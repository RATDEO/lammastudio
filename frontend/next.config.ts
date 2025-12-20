import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  images: { unoptimized: true },
  allowedDevOrigins: [
    'app.homelabai.org',
    'homelabai.org',
  ],
  async redirects() {
    return [
      {
        source: '/models',
        destination: '/recipes',
        permanent: true,
      },
    ];
  },
};

export default nextConfig;
