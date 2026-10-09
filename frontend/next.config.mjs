/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  webpack: (config, { isServer }) => {
    // Fix chunk loading when project path contains special chars (& and spaces)
    if (!isServer) {
      config.output = {
        ...config.output,
        publicPath: "/_next/",
      };
    }
    return config;
  },
};

export default nextConfig;
