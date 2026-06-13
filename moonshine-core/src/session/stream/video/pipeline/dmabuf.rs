//! DMA-BUF import support for zero-copy video encoding.
//!
//! This module provides the ability to import Linux DMA-BUF file descriptors as
//! Vulkan images for direct video encoding without CPU-side copies.
//!
//! `DmaBufImporter` caches imported Vulkan resources per DMA-BUF file descriptor
//! so that the same underlying buffer is imported only once. Subsequent frames
//! reusing the same DMA-BUF reuse the cached `VkImage` and `VkDeviceMemory`,
//! eliminating per-frame `vkCreateImage` + `vkAllocateMemory(DMA-BUF import)`
//! calls that can cost 0.5\u20131.5ms on NVIDIA drivers.
//!
//! Keying by `wl_buffer` ObjectId does not work: the NVIDIA Wayland WSI creates
//! a new `wl_buffer` wrapper each frame even though the underlying buffer is
//! stable, so ObjectId-keying would miss the cache every frame. The fd is
//! stable across those wrappers, but fd *numbers* are recycled for unrelated
//! buffers once closed (e.g. a swapchain recreation when a game switches HDR
//! format), so a different buffer can land an in-use cache slot. Validating the
//! import parameters on a hit rejects most of those, but a recycled fd that
//! happens to match the previous buffer's geometry would still be served a
//! stale `VkImage`, and the GPU then reads freed/foreign memory — a device-lost
//! fault.
//!
//! So the cache is keyed by the buffer's *inode* (`st_ino` of the DMA-BUF fd),
//! which identifies the underlying buffer object and is not recycled while the
//! buffer lives, so it cannot alias a different buffer the way an fd number
//! can. The import-parameter check is kept as defence-in-depth — and to catch
//! inode reuse after a buffer is freed and a new one allocated at the same
//! inode: a hit must match both the inode and the import parameters. Mismatched
//! entries are retired and freed after `CACHE_TTL`, once any in-flight GPU work
//! using them has completed.
//!
//! The cache evicts entries that have not been touched in `CACHE_TTL`. The
//! compositor holds client buffers alive until the encoder signals `consumed`,
//! so the fd is guaranteed valid during import. The 2s TTL ensures any
//! in-flight GPU work completes before cached Vulkan resources are freed.
//!
//! REQUIRES Linux >= 5.3. Per-buffer DMA-BUF inodes only exist from 5.3 (the
//! change that made `fstat` on a DMA-BUF meaningful); before that every
//! DMA-BUF shares one anonymous inode, so the key would collapse to a single
//! value and alias distinct buffers — strictly worse than fd-keying. The cache
//! cannot distinguish a genuine reuse from a shared-inode alias once it
//! happens, so this is guarded once up front: `DmaBufImporter::new` reads the
//! kernel version and warns loudly on a pre-5.3 host.

use ash::vk;
use pixelforge::VideoContext;
use std::collections::HashMap;
use std::os::fd::RawFd;
use std::os::unix::io::{BorrowedFd, IntoRawFd};
use std::time::{Duration, Instant};
use tracing::{debug, trace};

/// How long a cached import stays resident after its last use before being
/// evicted and freed. Long enough that any in-flight encoder/blitter work
/// using the image has definitely completed (depth-2 pipeline at 120 fps is
/// ~16 ms of in-flight latency), short enough that monotonically-growing
/// buffer-index churn doesn't accumulate VRAM.
const CACHE_TTL: Duration = Duration::from_secs(2);

/// Sweep for stale cache entries every N `import_or_reuse` calls. Cheap
/// (HashMap retain over a small map) but no point doing it every frame.
const SWEEP_INTERVAL_CALLS: u32 = 60;

/// `st_ino` of the DMA-BUF fd: the cache key. The inode identifies the
/// underlying buffer object and is stable across the per-frame `wl_buffer`
/// wrappers, while not aliasing a different buffer the way a recycled fd
/// number can. Plane 0's fd suffices; the per-hit parameter check catches any
/// mismatch.
fn dmabuf_inode(fd: RawFd) -> Result<u64, String> {
	let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
	// SAFETY: fstat writes the stat buffer for a valid fd; the return code is
	// checked before assuming the buffer is initialized.
	let rc = unsafe { libc::fstat(fd, stat.as_mut_ptr()) };
	if rc != 0 {
		return Err(format!(
			"fstat on DMA-BUF fd {fd} failed: {}",
			std::io::Error::last_os_error()
		));
	}
	Ok(unsafe { stat.assume_init() }.st_ino)
}

/// Warn once at startup if the kernel predates per-buffer DMA-BUF inodes
/// (Linux 5.3). On such a kernel every DMA-BUF shares one anonymous inode, so
/// the import cache's inode key collides across distinct buffers and may serve
/// the wrong image — a regression this cache cannot detect at runtime, hence
/// the up-front check. Fails open: an unparseable version is left un-warned
/// rather than producing a false alarm.
fn warn_if_dmabuf_inodes_unsupported() {
	let release = std::fs::read_to_string("/proc/sys/kernel/osrelease").unwrap_or_default();
	let mut parts = release.trim().split('.');
	let major: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
	let minor: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
	// major == 0 means we couldn't read/parse the version; don't warn on that.
	if major != 0 && (major, minor) < (5, 3) {
		tracing::warn!(
			"kernel {} predates Linux 5.3, where DMA-BUFs share a single inode: the zero-copy \
			 import cache keys collide across buffers and may serve the wrong frame (corruption \
			 or device-lost). Upgrade to Linux 5.3+ for DMA-BUF capture.",
			release.trim()
		);
	}
}

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

/// Identifying parameters of a DMA-BUF import. A cached image may only be
/// reused when all of these match the new request; a mismatch on the same inode
/// means the inode now maps to a different buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ImportParams {
	width: u32,
	height: u32,
	format: vk::Format,
	modifier: u64,
	plane_count: usize,
	/// (offset, stride) per plane.
	plane_layouts: [(u32, u32); 4],
}

impl ImportParams {
	fn new(width: u32, height: u32, format: vk::Format, planes: &[DmaBufPlane]) -> Self {
		let mut plane_layouts = [(0, 0); 4];
		for (i, p) in planes.iter().take(4).enumerate() {
			plane_layouts[i] = (p.offset, p.stride);
		}
		Self {
			width,
			height,
			format,
			modifier: planes.first().map_or(0, |p| p.modifier),
			plane_count: planes.len().min(4),
			plane_layouts,
		}
	}
}

/// Cached Vulkan resources for a single compositor buffer slot.
struct CachedImport {
	image: vk::Image,
	memory: vk::DeviceMemory,
	params: ImportParams,
	last_used: Instant,
}

/// Importer for DMA-BUF file descriptors into Vulkan images.
///
/// Owns a per-buffer cache of `VkImage` + `VkDeviceMemory` (keyed by inode)
/// with TTL eviction.
/// Layout transitions are deferred to the consumer (e.g. `ColorConverter`)
/// to avoid a separate GPU submission per first-time import.
pub(crate) struct DmaBufImporter {
	context: VideoContext,
	external_memory_fd: ash::khr::external_memory_fd::Device,
	/// Per-buffer cache — keyed by the DMA-BUF inode so the same underlying
	/// buffer is reused even when the driver wraps it in new `wl_buffer`
	/// objects, without a recycled fd number aliasing a different buffer.
	cache: HashMap<u64, CachedImport>,
	/// Imports whose inode now maps to a different buffer, awaiting TTL expiry
	/// before destruction (in-flight GPU work may still reference them).
	retired: Vec<CachedImport>,
	/// Calls since the last stale-entry sweep.
	calls_since_sweep: u32,
}

impl DmaBufImporter {
	/// Create a new DMA-BUF importer.
	pub fn new(context: VideoContext) -> Result<Self, String> {
		warn_if_dmabuf_inodes_unsupported();
		let external_memory_fd = ash::khr::external_memory_fd::Device::load(context.instance(), context.device());

		Ok(Self {
			context,
			external_memory_fd,
			cache: HashMap::new(),
			retired: Vec::new(),
			calls_since_sweep: 0,
		})
	}

	/// Import a DMA-BUF as a Vulkan image, reusing a cached import when
	/// the same DMA-BUF (by inode) has been seen before with the same parameters.
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
		fd: RawFd,
		width: u32,
		height: u32,
		format: vk::Format,
		planes: &[DmaBufPlane],
	) -> Result<(vk::Image, bool), String> {
		self.calls_since_sweep += 1;
		if self.calls_since_sweep >= SWEEP_INTERVAL_CALLS {
			self.calls_since_sweep = 0;
			self.evict_stale();
		}

		let params = ImportParams::new(width, height, format, planes);
		let inode = dmabuf_inode(fd)?;

		let now = Instant::now();
		if let Some(cached) = self.cache.get_mut(&inode) {
			if cached.params == params {
				cached.last_used = now;
				return Ok((cached.image, false));
			}
		}

		// A leftover entry whose parameters no longer match means this inode now
		// refers to a different buffer — e.g. the buffer was freed and its inode
		// reallocated, or the same buffer changed geometry (a swapchain
		// recreation when the game switches HDR format). Reusing the stale image
		// would make the GPU read the old buffer with the wrong format/stride —
		// a device-lost fault. Retire it for TTL-deferred destruction in case
		// prior GPU work still uses it.
		if let Some(mut stale) = self.cache.remove(&inode) {
			debug!(
				"inode {inode} now refers to a different buffer ({:?} -> {:?}); retiring stale import",
				stale.params, params
			);
			stale.last_used = now;
			self.retired.push(stale);
		}

		// First time seeing this DMA-BUF — full import.
		debug!(
			"First import for fd {fd}: {}x{}, format={:?}, stride={}, modifier={:#x}",
			width, height, format, planes[0].stride, planes[0].modifier
		);

		let (image, memory) = self.import_internal(width, height, format, planes)?;

		self.cache.insert(
			inode,
			CachedImport {
				image,
				memory,
				params,
				last_used: now,
			},
		);
		Ok((image, true))
	}

	/// Drop cached entries that haven't been touched in `CACHE_TTL` and free
	/// their backing Vulkan resources. Stale entries are guaranteed to be out
	/// of any encoder/blitter pipeline (TTL >> max in-flight depth at 120 fps),
	/// so it's safe to destroy without an explicit fence wait.
	fn evict_stale(&mut self) {
		let cutoff = Instant::now() - CACHE_TTL;
		let device = self.context.device();
		let destroy_if_expired = |v: &CachedImport| {
			if v.last_used < cutoff {
				unsafe {
					device.destroy_image(v.image, None);
					device.free_memory(v.memory, None);
				}
				false
			} else {
				true
			}
		};
		let before = self.cache.len() + self.retired.len();
		self.cache.retain(|_, v| destroy_if_expired(v));
		self.retired.retain(destroy_if_expired);
		let evicted = before - self.cache.len() - self.retired.len();
		if evicted > 0 {
			trace!(
				"DmaBufImporter: evicted {evicted} stale cache entries, {} live",
				self.cache.len()
			);
		}
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
			for cached in self.cache.drain().map(|(_, v)| v).chain(self.retired.drain(..)) {
				device.destroy_image(cached.image, None);
				device.free_memory(cached.memory, None);
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::{DmaBufPlane, ImportParams};
	use ash::vk;

	fn plane(offset: u32, stride: u32) -> DmaBufPlane {
		DmaBufPlane {
			fd: 42,
			offset,
			stride,
			modifier: 0xdead,
		}
	}

	#[test]
	fn import_params_match_for_identical_buffer() {
		let a = ImportParams::new(1920, 1080, vk::Format::A2B10G10R10_UNORM_PACK32, &[plane(0, 7680)]);
		let b = ImportParams::new(1920, 1080, vk::Format::A2B10G10R10_UNORM_PACK32, &[plane(0, 7680)]);
		assert_eq!(a, b);
	}

	#[test]
	fn import_params_detect_recycled_fd_after_format_switch() {
		// PQ→scRGB swapchain recreation: the new FP16 buffer can land on the
		// same fd number as the destroyed 10-bit buffer, with a different
		// format and stride. The cache must not treat this as a hit.
		let pq = ImportParams::new(1920, 1080, vk::Format::A2B10G10R10_UNORM_PACK32, &[plane(0, 7680)]);
		let scrgb = ImportParams::new(1920, 1080, vk::Format::R16G16B16A16_SFLOAT, &[plane(0, 15360)]);
		assert_ne!(pq, scrgb);
	}

	#[test]
	fn import_params_detect_stride_only_change() {
		let a = ImportParams::new(1920, 1080, vk::Format::R16G16B16A16_SFLOAT, &[plane(0, 15360)]);
		let b = ImportParams::new(1920, 1080, vk::Format::R16G16B16A16_SFLOAT, &[plane(0, 16384)]);
		assert_ne!(a, b);
	}
}
