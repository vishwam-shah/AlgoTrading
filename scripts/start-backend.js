const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const workspaceRoot = path.resolve(__dirname, "..");
const isWindows = process.platform === "win32";
const useReload = process.argv.includes("--reload");

function resolvePythonCommand() {
	const absoluteCandidates = isWindows
		? [
				path.join(workspaceRoot, "venv", "Scripts", "python.exe"),
				path.join(workspaceRoot, ".venv", "Scripts", "python.exe"),
			]
		: [
				path.join(workspaceRoot, "venv", "bin", "python3"),
				path.join(workspaceRoot, ".venv", "bin", "python3"),
			];

	for (const candidate of absoluteCandidates) {
		if (fs.existsSync(candidate)) {
			return { command: candidate, argsPrefix: [] };
		}
	}

	if (isWindows) {
		return { command: "py", argsPrefix: [] };
	}

	return { command: "python3", argsPrefix: [] };
}

const { command, argsPrefix } = resolvePythonCommand();
const port = process.env.BACKEND_PORT || "8000";
const pythonPath = process.env.PYTHONPATH
	? `${workspaceRoot}${path.delimiter}${process.env.PYTHONPATH}`
	: workspaceRoot;

const uvicornArgs = [
	...argsPrefix,
	"-m",
	"uvicorn",
	"backend.main:app",
	"--host",
	"0.0.0.0",
	"--port",
	port,
	"--log-level",
	"info",
];

if (useReload) {
	uvicornArgs.push("--reload", "--reload-dir", "backend");
}

console.log(
	`[backend] Launching with ${command} ${uvicornArgs.join(" ")} (cwd=${workspaceRoot})`
);

const child = spawn(command, uvicornArgs, {
	cwd: workspaceRoot,
	stdio: "inherit",
	env: {
		...process.env,
		PYTHONPATH: pythonPath,
	},
	shell: false,
});

child.on("exit", (code, signal) => {
	if (signal) {
		process.exit(1);
	}
	process.exit(code == null ? 1 : code);
});

child.on("error", (error) => {
	console.error("[backend] Failed to start backend process:", error.message);
	process.exit(1);
});
