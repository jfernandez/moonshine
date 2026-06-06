//! DMA-BUF import support for zero-copy video encoding.
//!
//! This module provides the ability to import Linux DMA-BUF file descriptors as
//! Vulkan images for direct video encoding without CPU-side copies.
//!
//! `DmaBufImporter` caches imported Vulkan resources per DMA-BUF, keyed by
//! the buffer's inode (`st_ino` of the plane-0 fd). The inode is stable
//! across `wl_buffer` re-wraps, so a client cycling a fixed pool of buffers
//! (mpv, gamescope) hits the cache even when it hands the compositor a fresh
//! `wl_buffer` every frame.
//!
//! Hits validate the import's geometry (size, format, modifier), so a
//! re-wrapped or (on older kernels) recycled inode is re-imported instead of
//! encoded from the wrong image. Entries unused for `CACHE_TTL` are evicted
//! a couple per call so destruction never bursts on the frame path, and
//! destroyed handles are reported via `take_destroyed` so the caller can
//! drop view caches keyed by them before Vulkan recycles a handle.

use ash::vk;
use pixelforge::VideoContext;
use std::collections::HashMap;
use std::os::fd::RawFd;
use std::os::unix::io::{BorrowedFd, IntoRawFd};
use std::time::{Duration, Instant};
use tracing::{debug, trace};

/// How long a cached import stays resident after its last use before being
/// evicted and freed. Long enough that any in-flight encoder/blitter work
/// using the image has definitely completed.
const CACHE_TTL: Duration = Duration::from_secs(2);

/// Evict at most this many stale entries per `import_or_reuse` call. Each
/// call inserts at most one entry and can retire two, so the cache stays
/// bounded while freeing device memory (a slow driver call) never bursts
/// on the frame path like a periodic sweep would.
const MAX_EVICTIONS_PER_CALL: usize = 2;

/// Information about a single DMA-BUF plane.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DmaBufPlane {
	/// File descriptor for the DMA-BUF.
	pub fd: RawFd,
	/// Offset within the DMA-BUF to the start of this plane.
	pub offset: u32,
	/// Row stride in bytes.
	pub stride: u32,
	/// DRM format modifier.
	pub modifier: u64,
}

/// Cached Vulkan resources for a single DMA-BUF.
struct CachedImport {
	image: vk::Image,
	memory: vk::DeviceMemory,
	last_used: Instant,
	/// Import geometry, validated on every cache hit.
	width: u32,
	height: u32,
	format: vk::Format,
	modifier: u64,
}

/// Importer for DMA-BUF file descriptors into Vulkan images.
///
/// Owns a per-buffer cache of `VkImage` + `VkDeviceMemory` with TTL
/// eviction. Layout transitions are deferred to the consumer
/// (e.g. `ColorConverter`/`RgbBlitter`) to avoid a separate GPU submission
/// per first-time import.
pub(crate) struct DmaBufImporter {
	context: VideoContext,
	external_memory_fd: ash::khr::external_memory_fd::Device,
	/// Keyed by the DMA-BUF's inode; the compositor's buffer index churns
	/// every frame for re-wrapping clients.
	cache: HashMap<u64, CachedImport>,
	/// Images destroyed since the caller last called `take_destroyed`.
	destroyed: Vec<vk::Image>,
}

/// Inode of the DMA-BUF behind `fd`, shared by every fd/`wl_buffer` that
/// wraps the same buffer object. Plane 0 suffices; the per-hit geometry
/// validation catches any mismatch.
fn dmabuf_inode(fd: RawFd) -> Result<u64, String> {
	let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
	// SAFETY: fstat writes the stat buffer for a valid fd; the return code
	// is checked before assuming initialization.
	let rc = unsafe { libc::fstat(fd, stat.as_mut_ptr()) };
	if rc != 0 {
		return Err(format!(
			"fstat on DMA-BUF fd {fd} failed: {}",
			std::io::Error::last_os_error()
		));
	}
	Ok(unsafe { stat.assume_init() }.st_ino)
}

impl DmaBufImporter {
	/// Create a new DMA-BUF importer.
	pub fn new(context: VideoContext) -> Result<Self, String> {
		let external_memory_fd = ash::khr::external_memory_fd::Device::load(context.instance(), context.device());

		Ok(Self {
			context,
			external_memory_fd,
			cache: HashMap::new(),
			destroyed: Vec::new(),
		})
	}

	/// Import a DMA-BUF as a Vulkan image, reusing a cached import when the
	/// same buffer (by inode) has been seen before. `buffer_index` is the
	/// compositor's per-`wl_buffer` index, used only for log correlation.
	///
	/// The `format` parameter specifies the Vulkan format matching the DMA-BUF
	/// pixel format (e.g. `B8G8R8A8_UNORM` for SDR, `A2B10G10R10_UNORM_PACK32`
	/// for 10-bit HDR, `R16G16B16A16_SFLOAT` for FP16 HDR).
	///
	/// Returns `(image, needs_transition)` where `needs_transition` is `true`
	/// for first-time imports whose image is still in `UNDEFINED` layout.
	/// The caller is responsible for transitioning the image (e.g. by passing
	/// the appropriate `src_layout` to `ColorConverter::convert`).
	pub fn import_or_reuse(
		&mut self,
		buffer_index: usize,
		width: u32,
		height: u32,
		format: vk::Format,
		planes: &[DmaBufPlane],
	) -> Result<(vk::Image, bool), String> {
		self.evict_stale();

		// An fstat failure drops this frame; the next one recovers.
		let inode = dmabuf_inode(planes[0].fd)?;
		let now = Instant::now();
		let modifier = planes[0].modifier;
		if let Some(cached) = self.cache.get_mut(&inode) {
			if cached.width == width && cached.height == height && cached.format == format && cached.modifier == modifier {
				cached.last_used = now;
				return Ok((cached.image, false));
			}
			// Same inode, different geometry: the client re-wrapped the same
			// buffer with different parameters (or an older kernel recycled
			// the inode). Destroy the stale import and import fresh.
			let stale = self
				.cache
				.remove(&inode)
				.expect("present: get_mut above found it");
			let device = self.context.device();
			unsafe {
				device.destroy_image(stale.image, None);
				device.free_memory(stale.memory, None);
			}
			self.destroyed.push(stale.image);
		}

		// First time seeing this buffer — full import.
		debug!(
			"First import for buffer {buffer_index} (inode {inode}): {}x{}, format={:?}, fd={}, stride={}, modifier={:#x}",
			width, height, format, planes[0].fd, planes[0].stride, planes[0].modifier
		);

		let (image, memory) = self.import_internal(width, height, format, planes)?;

		self.cache.insert(
			inode,
			CachedImport {
				image,
				memory,
				last_used: now,
				width,
				height,
				format,
				modifier,
			},
		);
		Ok((image, true))
	}

	/// Drop up to `MAX_EVICTIONS_PER_CALL` entries that haven't been touched
	/// in `CACHE_TTL` and free their backing Vulkan resources. Stale entries
	/// are long out of any encoder/blitter pipeline (TTL >> max in-flight
	/// depth), so no fence wait is needed.
	fn evict_stale(&mut self) {
		let cutoff = Instant::now() - CACHE_TTL;
		let mut victims: [Option<u64>; MAX_EVICTIONS_PER_CALL] = [None; MAX_EVICTIONS_PER_CALL];
		let mut found = 0;
		for (k, v) in &self.cache {
			if v.last_used < cutoff {
				victims[found] = Some(*k);
				found += 1;
				if found == MAX_EVICTIONS_PER_CALL {
					break;
				}
			}
		}
		if found == 0 {
			return;
		}
		let device = self.context.device();
		for key in victims.into_iter().flatten() {
			let v = self
				.cache
				.remove(&key)
				.expect("victim was collected from the map earlier this call");
			unsafe {
				device.destroy_image(v.image, None);
				device.free_memory(v.memory, None);
			}
			self.destroyed.push(v.image);
		}
		trace!(
			"DmaBufImporter: evicted {found} stale cache entries, {} live",
			self.cache.len()
		);
	}

	/// Images destroyed since the last call, for the caller to invalidate
	/// any handle-keyed view caches before Vulkan recycles the handles.
	pub fn take_destroyed(&mut self) -> Vec<vk::Image> {
		std::mem::take(&mut self.destroyed)
	}

	/// Perform the raw Vulkan import of a DMA-BUF with the specified format.
	///
	/// Returns the `(VkImage, VkDeviceMemory)` pair. The image is in
	/// `UNDEFINED` layout; the caller must transition it.
	fn import_internal(
		&self,
		width: u32,
		height: u32,
		format: vk::Format,
		planes: &[DmaBufPlane],
	) -> Result<(vk::Image, vk::DeviceMemory), String> {
		if planes.is_empty() {
			return Err("At least one DMA-BUF plane is required".to_string());
		}

		let device = self.context.device();

		// Build DRM format modifier plane layouts for all planes.
		// AMD modifiers (e.g. tiled/DCC) may require multiple planes;
		// the layout count must match the modifier's expected plane count.
		let plane_layouts: Vec<vk::SubresourceLayout> = planes
			.iter()
			.map(|p| {
				vk::SubresourceLayout::default()
					.offset(p.offset as u64)
					.row_pitch(p.stride as u64)
			})
			.collect();

		let modifier = planes[0].modifier;
		let mut drm_format_modifier_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
			.drm_format_modifier(modifier)
			.plane_layouts(&plane_layouts);

		let mut external_memory_info =
			vk::ExternalMemoryImageCreateInfo::default().handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
		external_memory_info.p_next =
			&mut drm_format_modifier_info as *mut vk::ImageDrmFormatModifierExplicitCreateInfoEXT as *mut _;

		let mut image_create_info = vk::ImageCreateInfo::default()
			.image_type(vk::ImageType::TYPE_2D)
			.format(format)
			.extent(vk::Extent3D {
				width,
				height,
				depth: 1,
			})
			.mip_levels(1)
			.array_layers(1)
			.samples(vk::SampleCountFlags::TYPE_1)
			.tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
			.usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED)
			.sharing_mode(vk::SharingMode::EXCLUSIVE)
			.initial_layout(vk::ImageLayout::UNDEFINED);
		image_create_info.p_next = &mut external_memory_info as *mut vk::ExternalMemoryImageCreateInfo as *mut _;

		let image = unsafe { device.create_image(&image_create_info, None) }
			.map_err(|e| format!("DMA-BUF image creation: {e}"))?;

		// Memory requirements.
		let mem_requirements = unsafe { device.get_image_memory_requirements(image) };

		// FD memory properties.
		let mut memory_fd_properties = vk::MemoryFdPropertiesKHR::default();
		unsafe {
			self.external_memory_fd.get_memory_fd_properties(
				vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
				planes[0].fd,
				&mut memory_fd_properties,
			)
		}
		.map_err(|e| format!("Failed to get memory FD properties: {e}"))?;

		// Duplicate the FD — vkAllocateMemory consumes it.
		let fd = unsafe { BorrowedFd::borrow_raw(planes[0].fd) }
			.try_clone_to_owned()
			.map_err(|e| format!("Failed to duplicate DMA-BUF FD: {e}"))?
			.into_raw_fd();

		let mut import_memory_fd_info = vk::ImportMemoryFdInfoKHR::default()
			.handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
			.fd(fd);

		let memory_type_bits = mem_requirements.memory_type_bits & memory_fd_properties.memory_type_bits;

		debug!(
			"Memory allocation: size={}, image_type_bits={:#x}, fd_type_bits={:#x}, combined={:#x}",
			mem_requirements.size,
			mem_requirements.memory_type_bits,
			memory_fd_properties.memory_type_bits,
			memory_type_bits
		);

		let memory_type_index = self
			.context
			.find_memory_type(memory_type_bits, vk::MemoryPropertyFlags::empty())
			.ok_or_else(|| "No suitable memory type for DMA-BUF import".to_string())?;

		// Dedicated allocation (required by many drivers for external memory).
		let mut dedicated_alloc_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
		import_memory_fd_info.p_next = &mut dedicated_alloc_info as *mut vk::MemoryDedicatedAllocateInfo as *mut _;

		let mut alloc_info = vk::MemoryAllocateInfo::default()
			.allocation_size(mem_requirements.size)
			.memory_type_index(memory_type_index);
		alloc_info.p_next = &mut import_memory_fd_info as *mut vk::ImportMemoryFdInfoKHR as *mut _;

		let memory = unsafe { device.allocate_memory(&alloc_info, None) }.map_err(|e| {
			unsafe { device.destroy_image(image, None) };
			format!("DMA-BUF memory import: {e}")
		})?;

		if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
			unsafe {
				device.free_memory(memory, None);
				device.destroy_image(image, None);
			}
			return Err(format!("DMA-BUF memory bind: {e}"));
		}

		Ok((image, memory))
	}
}

impl Drop for DmaBufImporter {
	fn drop(&mut self) {
		let device = self.context.device();
		unsafe {
			for (_, cached) in self.cache.drain() {
				device.destroy_image(cached.image, None);
				device.free_memory(cached.memory, None);
			}
		}
	}
}
