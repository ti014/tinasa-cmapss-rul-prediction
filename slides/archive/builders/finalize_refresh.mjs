import path from "node:path";
import { pathToFileURL } from "node:url";

const workspaceDir = process.env.WORKSPACE_DIR ?? "D:/IUH/BigData/Final_Project";
const skillDir = process.env.SKILL_DIR ?? "C:/Users/timewo/.codex/plugins/cache/openai-primary-runtime/presentations/26.923.10815/skills/presentations";
const candidatePath = process.env.CANDIDATE_PPTX ?? path.join(workspaceDir, "tmp", "slides-refresh-20260924", "candidate.pptx");
const finalPath = process.env.FINAL_PPTX ?? path.join(workspaceDir, "slides", "cmapss_fd001_presentation_updated.pptx");
const stagingDir = path.join(workspaceDir, "tmp", "slides-refresh-20260924", "finalizer");
const runtimePython = process.env.RUNTIME_PYTHON ?? "C:/Users/timewo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const runtimeNodeModules = process.env.RUNTIME_NODE_MODULES ?? "C:/Users/timewo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";

const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools", "artifact_tool_utils.mjs")).href);
const result = await finalizePresentation({
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    "--require-native-table-slide", "6",
    "--require-native-table-slide", "8",
  ],
  requiredNativeTableOwnerSlides: [6, 8],
  sourceTemplatePath: path.join(workspaceDir, "slides", "archive", "decks", "cmapss_fd001_presentation_before_refresh.pptx"),
  fontPolicy: {
    basis: "reference",
    families: ["Arial"],
    referencePath: path.join(workspaceDir, "slides", "archive", "decks", "cmapss_fd001_presentation_before_refresh.pptx"),
    referenceSha256: "191a2f4ee9be11d896ea557babbd8c388056352c128529e3c0dad8b89e83d22b",
  },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "cmapss_fd001_presentation_updated.pptx.validation.json"),
});
console.log(JSON.stringify(result, null, 2));
