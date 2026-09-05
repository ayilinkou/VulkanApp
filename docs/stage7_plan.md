# Stage 7 — Engine shell & dependency injection: settled design

**This document is temporary. Delete it when Stage 7 completes.** It records the decisions taken
before the stage started — in a `/grill-me` interview on 2026-08-30 and 2026-08-31 — so that the
work does not re-litigate them a step at a time. Nothing here is a new step: Part IV's steps 40b
and 41–47 remain the work order, and this says how each of them is to be built.

Unlike `rhi_extraction_plan.md`, which was kept past its stage because its decisions still govern a
seam that outlived it, this one has no such afterlife. Anything still worth knowing when the stage
ends belongs in the code it describes, or in `architecture_plan.md`; the last section lists the
amendments that document owes.

**Where this and Part IV disagree, this wins** — it was written later and with the code open. Where
this and `rhi_extraction_plan.md`'s D0–D12 disagree about the RHI's public seam, the RHI plan wins,
as it does everywhere else.

---

## Order of work

**41 → 42 → 43 → 44 → 45 → 46 → 47, then 40b.**

Part IV lists 40b first because it arrived from Stage 6, not because anything depends on it.
Nothing in 41–47 needs the event seam except 40b's own verification: `--frames N` with a capture
requires no input at all. Running it last puts the CI checkpoint — the stage's stated goal — as
early as the dependencies allow, and keeps 40b's subtree (the event abstraction, the scripted-input
format, `HeadlessPlatform`'s queue) out of the way until then.

## Cross-cutting

**A module is created only where a step's own deliverable is that module's type.** `engine/assets`
appears at step 42 because `AssetRegistry` is an assets type; `engine/editor` at 46 because the
editor is. Renderer internals stay in `engine/engine` until Stage 8 draws the Render/Scene boundary
deliberately — moving them earlier would be drawing that boundary by accident.

**The module stays named `Render`, not `Renderer`.** Every other module is a domain noun, and
`Renderer` yields `Hikari::Renderer::Renderer`.

---

## 41. `Engine` + `EngineConfig` + `RunSpec` + `RunReport`

- **`RunSpec` is engine-scoped.** Apps parse their own platform flags into a `WindowDesc`; the
  struct handed to `Engine::Run` carries nothing the engine does not read. A headless binary has no
  business describing a window mode to something that cannot open one.
- **A shared `ParseRunSpec` lives in `engine/engine`**, beside `RunSpec`. Each app adds its own
  platform flags and its own help section on top, so the two binaries cannot drift on the flags
  they share.
- **`Engine::Run` returns data and writes nothing** — a `RunReport` plus a `CapturedFrame`. The apps
  write files. `RunSpec` therefore carries `bCaptureFinalFrame`, not paths: a test that wants the
  pixels should not have to give the engine a filename and read them back off disk.
- **`WriteRunReport` lives in `engine/engine`; image encoding in `engine/assets`**, which already
  owns stb for decoding. Both stay in `src/main.cpp` until the step that moves them.
- **`EngineConfig` is a construction-time input, separate from `RunSpec`.** Its values size GPU
  resources once, at startup, where `RunSpec`'s describe a run. Only `--frames-in-flight` gets a
  flag; `INITIAL_INSTANCE_CAPACITY` and `SKY_COLOR` are defaults with no CLI surface until
  something needs them to have one.
- **ImGui is told `MinImageCount = 2` and `ImageCount = max(2u, GetImageCount())`.** The vendored
  ImGui asserts `MinImageCount >= 2` and `ImageCount >= MinImageCount`
  (`imgui_impl_vulkan.cpp:1298-1299`), and an `OffscreenTarget` makes one image per frame in
  flight — so this step's own verify, `--frames-in-flight 1`, would violate it. `ImageCount` sizes
  only ImGui's private vertex/index ring (`:547`) and its unused-texture delay (`:899`), so
  over-provisioning it is safe and under-provisioning is the hazard. The engine's real frame count
  is unaffected, and `--frames-in-flight 1` keeps testing the one-image path it was added to test.
  Handle it deliberately rather than leaving it: `IM_ASSERT` is `assert`, so the violation aborts
  in Debug and passes silently in Release.

## 42. Inject `ResourceManager` and the loaders

- **Minimal, and deliberately so: de-singleton in place.** `AssetRegistry` is today's
  `ResourceManager` without `Get()` — still path-keyed, still owning the loaders. The
  `AssetId`/`*Data` redesign is Stage 9's, and doing it here would hide a dependency-injection
  change inside a data-model change.
- **`LoadScope`'s batching invariant must survive the move.** Injecting an `IUploadContext&` into
  each loader would undo it silently — every loader would flush its own uploads and the batching
  would be gone with nothing failing.
- **That batching gets a test**, and `IUploadContext` exposes its submission count so the test has
  something to assert on. The count also goes into the run report.
- **The report's `counters` splits by scope**: `counters.frame` for last-frame values,
  `counters.run` for cumulative ones, including upload submissions. This moves the committed
  baseline — bring the diff for approval rather than promoting it.

## 43. Inject `MaterialFactory`

- **The engine owns it, as a sibling of `AssetRegistry`, injected into the loaders that need it.**
  This overrides Part IV's "owned by `AssetRegistry`".

  The factory has callers on both sides of the asset/renderer line: `ModelLoader.cpp:96` calls
  `CreatePBRMaterial` while loading, and `main.cpp:1601` and `:1660` call
  `GetDescriptorSetLayout()` to build the opaque and transparent pipeline layouts. A descriptor set
  layout is a renderer input, and registry ownership would make the asset layer the thing that
  hands it out — putting a `vk::DescriptorSetLayout` on `engine/assets`' public surface and adding
  an `rhi/vulkan/` allowlist entry to a module that otherwise needs none. That list is a ratchet
  meant to shrink.
- **It stays in `src/`.** It is not an assets type, so nothing about this step requires it to move;
  Stage 8 relocates materials and pipelines together, which is when the real question — materials
  versus pipeline layouts — has to be answered anyway.
- Its inputs are `Rhi::IDevice&` and the texture sampler, both engine-owned, so `Engine`'s
  constructor builds the factory before the registry.

## 44. Inject `ModelManager` and break `Model`'s back-pointer

- **`CollectRenderables` runs every frame**, matching what `GenerateBatches` does today. Dirty-flag
  invalidation is a Stage 9 optimisation, and adding it here would make an ownership change also a
  behaviour change.

## 45. Deterministic clock

- **The step shrinks to `IClock` only** — for the simulation timestep. The measurement clock stays
  real.
- **No forcing of `SerialJobSystem`:** nothing today can make job order affect output, and
  `--jobs 0` already exists for diagnosis if that changes.
- **No `--seed`** until something is actually random.

## 46. `apps/` split

- **`--headless` is deleted; the binary is the mode.** The "cannot be combined with" checks against
  `--borderless` and `--fullscreen` go with it — they exist only because one binary had to refuse
  its own flags.
- **The two apps are `HikariEditor` (`apps/editor/`) and `HikariHeadless` (`apps/headless/`).**
  The windowed application is the editor, and naming it so is worth the rename. `HikariEngine`
  survives wherever the string names the *product* rather than the binary: `Paths.cpp`'s `kAppName`
  (a user-data directory), `DeviceDesc`'s default `ApplicationName`, the repository, README and
  docs. What moves with the target: `cmake/Shaders.cmake:103`'s `add_dependencies`,
  `tests/scripts/baseline_test.sh:39` and `.bat:31`, `CLAUDE.md`'s run examples, and CI. Both apps
  keep writing to `HIKARI_EXE_DIR`, so shaders still compile once beside both.
- **`HikariHeadless` links `Editor` and attaches it.** §8's target table forbids this; the table is
  amended. Headless means no window, not a feature-reduced build, and step 40a's coverage argument
  — that a headless CI run is the only place ImGui's bring-up and drawing get exercised
  automatically — does not weaken just because the binary split. Both apps link SDL through
  `Platform` regardless, so nothing is saved by refusing.
- **`engine/editor` holds the whole ImGui stack**, behind an `IUiBackend` interface —
  `Init`/`Shutdown`/`NewFrame`/`Render(ImDrawData*, ICommandList&)` — with `VulkanUiBackend` as its
  only implementation today. `EditorLayer` and the panels never name a graphics API, so a D3D12
  backend is a sibling file rather than an edit, and the eventual option of rendering ImGui through
  the RHI itself becomes a third implementation of the same interface rather than a rewrite. This
  is D9's position kept intact — the ImGui integration is backend-specific code and stays so — with
  the backend-specific part demoted to a leaf. `VulkanNative.h`'s allowlist entry moves from
  `src/main.cpp` to that backend file. The platform half needs no abstraction: ImGui's SDL3 backend
  already has `InitForVulkan`, `InitForD3D` and `InitForOther`.
- **The pixel comparison still passes `--no-ui` on both sides**, for the reason step 46 already
  gives: a windowed run carries a hover highlight that no headless run can reproduce.

## 47. Wire headless tests into CI

### What the tests are

- **Both kinds, split by what each is good at.** The scene matrix runs **in-process**: Catch2 links
  `Engine`, builds a `RunSpec`, calls `Engine::Run` and asserts on the returned `RunReport` — no
  file, no JSON parser, every counter assertable, and a failure that points at a line. One
  **subprocess** smoke test launches the real `HikariHeadless` and asserts its exit code and that
  the artefacts appeared, which is the only way to cover argument parsing, `main`'s wiring, report
  and screenshot writing. `--strict-validation` already exists (`main.cpp:208`, `:426`, `:3081`),
  so the exit code carries the validation result.
- **The in-process matrix attaches an `EditorLayer`.** Its `barriers` and `barrierCalls` then
  describe the frame that actually ships, rather than a composition nothing runs. Panel content
  cannot perturb the other expectations: `drawCalls` and `batches` count only scene batches
  (`main.cpp:1034-1036`, set at `:1964` and `:2053`).

### What the tests load

- **Two committed cubes, `cube.gltf` and `cube_transparent.gltf`**, each a single-file glTF with
  the buffer base64-embedded — JSON, so ~2 KB of reviewable text with no `.bin` sidecar and no
  binary blob, exercising the same loader path real content uses.
- **Material factors only, no textures.** `PBRMaterial::LoadTextures` falls back to
  `AI_MATKEY_BASE_COLOR` and the material bindings are `ePartiallyBound`
  (`MaterialFactory.cpp:70-73`), so an untextured cube renders. Transparency needs nothing else
  either: `Material::DetectBlendMode` calls a material transparent when opacity or base-colour
  alpha is below 1, so `baseColorFactor: [r, g, b, 0.5]` is the whole mechanism. The textured path
  is covered by the `test_scene.map` case, against far more representative content.
- `suggested_work.md` §1.6's null-material P0 is not in the way: its trigger is a mesh present in
  the file but unreferenced by the node graph, which a hand-authored cube does not have.
- **Scenes:** `empty`, `single_cube`, `two_materials`, `lights_only`, `transparent_only`,
  `instanced_cubes` — two entities of the same cube, so 1 batch and 2 instances, the only case that
  constrains `instances` separately from `batches` — plus one case running
  `content/scenes/test_scene.map`. The cube scenes exist so the expected counters are *derivable*;
  the real-content case exists so the matrix is not exclusively testing geometry nobody ships.
  `sponza` is gitignored, so the two Sponza scenes cannot run in CI at all.
- **Test data lives under `tests/data/` as its own content root**, located by
  `tests/support/TestPaths.h`. `Paths` is a constructed object rather than a global
  (`platform/Paths.h:45`), so each case resolves the root it needs and the `test_scene.map` case
  simply points at the repository's `content/`.
- **They carry a new `scene` CTest label**, and `scripts/precommit.sh` runs it. `ctest -L gpu`
  keeps its under-60-second meaning, a red step says which kind of thing broke, and precommit stays
  the strict superset of CI that `CLAUDE.md` promises.

### What CI runs

- **Scene tests are steps inside the existing ubuntu matrix jobs, not new jobs.** The expensive
  part of this CI is the build; a dedicated job pays it twice. `static-checks` is not a precedent
  for splitting — it was split because those checks are fast *and* report when the build is broken,
  and neither half transfers to a test that requires the build. lavapipe (`mesa-vulkan-drivers`)
  installs on the ubuntu runners.
- **Both `ninja-debug-linux` and `ninja-asan-linux` run every scene**, including `test_scene.map`.
  ASan is where a leak in texture loading or model teardown surfaces, and `test_scene.map` is the
  only case that exercises it. If it proves too slow, the lever is that one case on that one job,
  pulled from a measured runtime rather than a guess.
- **The non-ASan host is `ninja-debug-linux`, not release**, because `main.cpp:167` gates validation
  on `NDEBUG`: a Release build reports zero validation errors trivially, which would make step 47's
  headline assertion theatre. Making validation runtime-selectable is a backlog row; if it lands,
  promoting the scene step to release is a two-line CI change.
- **`ctest -L gpu` runs on all three ubuntu jobs, release included.** The GPU tests do not go
  through that `NDEBUG` gate — `tests/support/RhiTestFixture.h:100` sets `bEnableValidation = true`
  unconditionally — so a release run asserts everything a debug one does, and release is the only
  configuration in the matrix that is structurally different, since ASan inherits debug. The
  asymmetry with the scene steps is deliberate and belongs in a comment in the workflow.
- **The determinism check is a self-consistency double run on the same runner**, on
  `ninja-debug-linux`, comparing the report's counters **and** the decoded pixels on every scene
  including `test_scene.map`. A committed lavapipe baseline is deferred: comparing pixels across
  device classes is not a valid check.

  The caveat, which belongs in a comment next to the test: `Drawable::operator<`
  (`src/Drawable.h:16-22`) falls through to comparing `pMesh` and `pMat` by pointer value, so batch
  order tracks heap addresses. Opaque geometry is unaffected — that pipeline has
  `blendEnable = vk::False` (`main.cpp:1596`) and depth testing makes it order-independent. WBOIT
  accumulates additively (`main.cpp:1645-1650`), and float addition is commutative but not
  associative, so **three or more** stacked transparent layers over one pixel can differ in the low
  bits; two cannot. `test_scene.map`'s car glass can reach three. If that ever fires, it is a known
  cause rather than a regression, and step 58's ordering fix removes it.
- **Device selection stays `VK_DRIVER_FILES`** to pin the ICD in CI. `--gpu <substring>` waits for a
  real case and is tracked in `backlog.md`; enumeration order is not a stable identifier, so a bare
  index is not the answer.

## 40b. Event seam + scripted input — last

Deferred to the end of the stage together with everything under it: the event abstraction, the
scripted-input format, and `HeadlessPlatform`'s in-memory queue. It blocks only the scripted half
of step 47's tests, and none of the tests above need input.

---

## Deferred, and what each blocks

| Deferred | Until | Blocks |
|---|---|---|
| Event seam, scripted input, headless input queue | step 40b, end of stage | the scripted half of 47's tests, nothing else |
| `AssetId` / `*Data` redesign | Stage 9 | nothing in this stage |
| Dirty-flag invalidation for `CollectRenderables` | Stage 9 | nothing; step 44 recollects every frame |
| Runtime-selectable validation layers | backlog | a release scene step |
| `--gpu <substring>` device selection | backlog | nothing; CI pins the ICD instead |
| A committed lavapipe reference image | needs same-device comparison | cross-machine pixel checks |
| `Drawable`'s reproducible ordering | step 58 | pixel-exactness of 3+ transparent layers |

## Amendments owed to `architecture_plan.md`

Make each of these in the step that earns it, not up front:

- **§8's target table:** the `HikariEngine` / `HikariEngineHeadless` rows become `HikariEditor` /
  `HikariHeadless`, and the headless row's "must NOT know about" column loses `ImGui` and `Editor`.
- **Step 43:** "owned by `AssetRegistry`" becomes engine-owned and injected.
- **Steps 46 and 47:** the binary names in the Do and Verify text.
- **Step 40a's ImGui section** still claims the panel is in the committed baseline screenshot;
  `tests/scripts/baseline_test.sh:40` passes `--no-ui`, so it is not.
- **The stage table in `CLAUDE.md`** moves to **Stage 7.5**, not Stage 8, when this stage
  completes — `docs/backend_readiness_plan.md` inserts B1–B6 between the two — and this
  document is deleted in the same commit.
