/** Static export: the FastAPI backend serves the built files, so the demo needs only Python at runtime. */
const nextConfig = { output: "export", trailingSlash: true, images: { unoptimized: true } };
export default nextConfig;
