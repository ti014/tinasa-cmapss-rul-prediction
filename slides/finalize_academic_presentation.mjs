import path from "node:path";
import fs from "node:fs/promises";
import { pathToFileURL } from "node:url";

const workspaceDir = "D:/IUH/BigData/Final_Project";
const skillDir = "C:/Users/timewo/.codex/plugins/cache/openai-primary-runtime/presentations/26.923.10815/skills/presentations";
const candidatePath = path.join(workspaceDir, "tmp", "slides-transfer-20260925", "candidate.pptx");
const finalPath = path.join(workspaceDir, "slides", "cmapss_fd001_presentation.pptx");
const stagingDir = path.join(workspaceDir, "tmp", "slides-transfer-20260925", "finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools", "artifact_tool_utils.mjs")).href);

const result = await finalizePresentation({
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: "C:/Users/timewo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
  integrityValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    ...[3, 6, 7, 8, 11, 15].flatMap((n) => ["--require-native-table-slide", String(n)]),
  ],
  requiredNativeTableOwnerSlides: [3, 6, 7, 8, 11, 15],
  sourceTemplatePath: path.join(workspaceDir, "slides", "archive", "decks", "cmapss_fd001_presentation_before_refresh.pptx"),
  fontPolicy: {
    basis: "reference",
    families: ["Arial"],
    referencePath: path.join(workspaceDir, "slides", "archive", "decks", "cmapss_fd001_presentation_before_refresh.pptx"),
    referenceSha256: "191a2f4ee9be11d896ea557babbd8c388056352c128529e3c0dad8b89e83d22b",
  },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "cmapss_fd001_presentation.pptx.validation.json"),
});
console.log(JSON.stringify(result, null, 2));
