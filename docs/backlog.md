# HikariEngine — Backlog

Work that is off the critical path. The architecture plan is what happens next; this is
everything else, and it is deliberately broad — an item belongs here whether or not it can be
picked up today.

**Priority** is P1 (most important), P2, P3, and it means *how soon this should happen*, not
how broken the thing is. It is a different axis from `suggested_work.md`'s P0–P3, which is
severity assigned once by a review — a P2 design-debt item there can perfectly well be a P1
here. Both documents name their axis, "Severity tags" there and `Priority` here, because the
letters overlap and the scales do not.

**Blocked by** is blank for most rows, and a blank means what this list used to promise for
every row: pick it up any time. Where it names something, the item is still worth recording
here rather than buried in the step that unblocks it — the frame-time defect sat in the
architecture plan's §14 prose through five stages before anyone tracked it.

Each item is verified by "output unchanged unless noted, zero validation errors". Completed
items are deleted rather than struck through, along with any expanded note below the table;
git history is the record.

| Priority | Item | Where | Size | Blocked by |
|---|---|---|---|---|
| P1 | Correctness fixes from `suggested_work.md` §1.6 and §3.1 — §3.2 (batched uploads) is done | various | S–M each | |
| P1 | Restore `validate_best_practices` in `VulkanDevice.cpp`, commented out on 2026-09-05. vulkan-validationlayers 1.4.357.0 reads an image's last-used queue family in a maintenance9-gated branch of `BestPractices::ValidateImageInQueue` without checking it against `VK_QUEUE_FAMILY_IGNORED`, so the first use of any image in a submit segfaults inside the layer — every Debug run and two GPU tests. Fixed upstream by Vulkan-ValidationLayers PR #12922, merged after the 1.4.357.0 tag was cut, so no version vcpkg offers yet contains it | `VulkanDevice.cpp`, `vcpkg-configuration.json` | XS | a vcpkg baseline offering vulkan-validationlayers newer than 1.4.357.0 |
| P2 | One capture per run, and a name that cannot hold two. `DrawFrame` stages a capture only while `m_bScreenshotBufferReady` is false, so a script asking for `screenshot` twice gets one file and no warning about the other; `Engine::Run` returns a single `CapturedFrame` and the app writes it once. The naming compounds it: `GenerateTimestamp()` is second-resolution and the PNG and the report share one stamp, so two runs a second apart overwrite each other, and per-capture files would collide the moment more than one is written. Wants a captures list keyed by frame, a name that includes the frame, and a dropped request that says so | `Engine.cpp`, `RunApp.cpp` | M | |
| P2 | `--present-mode <immediate\|mailbox\|fifo\|fifo-relaxed>`, defaulting to mailbox; an explicit mode that the surface does not offer is a hard error | `rhi/IPresentTarget.h`, `SwapchainUtil.h`, `main.cpp` | S | |
| P2 | Document the matrix convention once and apply it consistently | `opaque.slang` header comment | S | |
| P2 | `.map` format `version` attribute | `XmlParser` | XS | |
| P2 | Record the GPU name, driver version, Vulkan API version, OS and architecture in the run report — two reports from different machines are otherwise comparable-looking and not comparable | `main.cpp`, `rhi/IDevice.h` | S | a neutral device-info accessor on `IDevice`, which is a seam decision — so Stage 7.5, which is where seam decisions are taken. Two backends make "which device produced this report" worth more, not less |
| P1 | A baseline comparison script — decode both PNGs, report the diff bounding box, and diff the report's `counters`. Today `CLAUDE.md` has to tell a human to drive PIL by hand, and that is not theoretical: the documented recipe compared RGBA images with `getbbox()`, which defaults to `alpha_only=True` and therefore inspected only the alpha channel. Every capture is fully opaque, so the check passed for any two images at all until it was corrected on 2026-09-05 | `tests/scripts/` | S | |
| P2 | Make the validation layers runtime-selectable instead of `NDEBUG`-gated. `DeviceDesc::bEnableValidation` is already a runtime field, but `main.cpp:167` decides it at compile time, so a Release build reports zero validation errors trivially and cannot assert them | `main.cpp`, later `RunSpec` | S | |
| P2 | A Debug build cannot start where `VK_LAYER_KHRONOS_validation` is not installed: validation goes into `requiredLayers`, so `VulkanDevice.cpp:604` throws `Required layer not supported` at instance creation rather than logging and continuing without it. Distinct from making validation runtime-selectable — this is what should happen when the layer is simply absent, as it is on a fresh clone without the SDK's layers | `VulkanDevice.cpp` | S | |
| P2 | Namespace `src/`'s remaining types under `Hikari::` | `src/` | M | Stages 7–9, which move them into engine modules a piece at a time |
| P2 | An engine config file feeding `EngineConfig`, with flags overriding it. Format, precedence, where the file is found, and how the effective values reach the run report are all open — design it when a setting actually needs persisting, rather than defaulting into a shape | `engine/engine` | M | `EngineConfig` existing (step 41) |
| P2 | The GPU tests assert a synchronization dependency they cannot detect: with sync validation off, dropping the wait semaphore from an offscreen read still passes | `tests/gpu/`, `rhi/DeviceDesc.h` | M | |
| P3 | `--gpu <name-substring>` device selection. Device creation takes the first suitable device in enumeration order, so a machine with both lavapipe and a real GPU gets whichever the loader lists first; CI pins the ICD with `VK_DRIVER_FILES` instead. Enumeration order is not a stable identifier, so a bare index is not the answer | `rhi/DeviceDesc.h`, `VulkanDevice.cpp` | S–M | |
| P3 | The ImGui panel has no regression coverage: the baseline is captured with `--no-ui`, deliberately, because a UI capture's hover highlight follows wherever the mouse was left | `tests/`, editor | M | Stage 7's `EditorLayer`, which can be driven without a mouse |
| P3 | Expose cloud push-constants in ImGui (`m_CloudData` is pushed but never written) | `CloudSystem` + editor UI | S | |
| P3 | `surface.slangh` to de-duplicate ~130 lines across the two surface shaders | `shaders/` | M | |
| P3 | Split `pbr.slangh` into `brdf`/`tonemap`/`phase` | `shaders/` | S | |
| P3 | `CubemapCreateInfo` → `std::array<std::string,6> FacePaths`, delete the 6-case switch | `CubemapLoader.cpp` | S | |
| P3 | Finish the skybox (loaded at `main.cpp:598`, never rendered) and reuse it for IBL | new pass | M–L | |

Two of these are worth expanding on, because they are latent defects or carry a decision:

- **The synchronization the GPU tests do not check.** `tests/gpu/rhi/PresentTargetTests.cpp`
  reads an offscreen image after rendering into it, and orders that copy after the render by
  waiting on the target's render-complete semaphore — passed in explicitly so that the wait is
  visible at the call site rather than hidden in a helper. The comment above the case says the
  ordering "is ordering it establishes", and that a stray `WaitIdle` would hide a read that
  established none.

  **It does not currently establish anything the test can see.** Dropping the semaphore and
  re-running leaves every case passing. Nothing catches it, for two reasons that compound:
  standard validation does not track memory hazards at all, and synchronization validation —
  which does — is off by default. What is left is the driver's own timing, which on this
  machine happens to finish the render before the copy starts.

  That is the worst shape a test can have. It reads as covering the hazard, so a future change
  that drops the wait, narrows the barrier's source stage below the semaphore's signal stage,
  or reorders the submit will land green; and a missing dependency does not fail where it was
  written. It fails intermittently, on another driver, in whatever runs next — which is the
  most expensive kind of graphics bug precisely because the evidence is nowhere near the cause.
  This is also the one hazard class the RHI's own design leans on tests to check, since the
  frame loop hands out semaphores that callers are trusted to wait on.

  Fixing it means running the `gpu`-labelled tests with synchronization validation enabled,
  which needs a way to ask for it: today `DeviceDesc` carries `bEnableValidation` and nothing
  finer, so the knob is a new field (or a validation-features struct) plumbed to the instance's
  `VkValidationFeaturesEXT` / `VK_EXT_layer_settings` chain. The check that it worked is the
  experiment above run in reverse — delete the wait, and require that the suite now fails.

- **`--present-mode`, and why the two failure policies differ.** The default stays what it is
  today: prefer mailbox, fall back to FIFO. **An explicitly requested mode that the surface
  does not offer is a hard error naming what was asked for and listing what is available** —
  never a silent downgrade. The whole reason to pass the flag is to test a specific mode, and
  a run that quietly measured a different one is worse than a run that refused: it produces a
  number that looks valid and is not.

  That is deliberately the opposite policy from `DeviceDesc::DisabledOptionalExtensions`,
  which reports and ignores a name it does not recognise. The cases differ: disabling an
  extension that was never present still achieves the intent, whereas asking for immediate
  and getting FIFO means the measurement is of something else.

  Two constraints on the implementation. **The default must stay a preference**, because only
  FIFO is guaranteed by the spec — mailbox is not, and a strict default would refuse to launch
  on a surface without it. And the *mode* is neutral vocabulary under D13 ("where only one API
  has the concept at all, its term stands"): Vulkan names these, D3D12 spells the same
  behaviour as `SyncInterval` plus `ALLOW_TEARING`, so this is `--present-mode` rather than
  `--vk-present-mode`.

  Reject `--present-mode` together with `--headless`, alongside the borderless/fullscreen
  check step 40a adds — an offscreen target does not present, so there is no mode to choose.

  **Log the mode that was actually chosen**, so a fallback is visible rather than inferred.
  The place for it is the existing one-line summary at the end of `SwapchainTarget::Create` —
  `"Swapchain: {}x{}, {} images"` — which becomes `"Swapchain: {}x{}, {} images, {}"`. Not
  surface creation: the surface exists before any mode is chosen, and `ChoosePresentMode` runs
  against `getSurfacePresentModesKHR` during swapchain creation, so the surface has nothing to
  report yet. `Create` is also called from `Recreate`, so the line already fires on every
  resize and fullscreen toggle and already carries an extent that changes each time — the mode
  rides along at no extra noise, and a mode that changed across a recreate shows up without a
  second log site or a "did it change" comparison.

  That one line covers both paths. An explicit mode that is unavailable throws before this
  point, naming what the surface offers; the default path cannot throw, so printing what it
  settled on is the only way a mailbox→FIFO fallback is ever visible.

  Worth pairing with the frame-time fix above: once the report carries real wall-clock
  timings, it should also carry the present mode, because two reports taken under different
  modes are not comparable.
