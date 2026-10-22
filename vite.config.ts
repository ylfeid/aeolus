import { defineConfig } from 'vite';

// Relative base so the build works under a GitHub Pages subpath.
export default defineConfig({
  base: './',
  build: { target: 'es2022' },
});
