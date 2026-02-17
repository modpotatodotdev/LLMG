import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://modpotatodotdev.github.io',
  base: '/LLMG',
  integrations: [
    starlight({
      title: 'LLMG',
      description: 'High-performance Rust LLM Gateway — unified API for 70+ providers',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/modpotatodotdev/LLMG' },
      ],
      head: [
        {
          tag: 'meta',
          attrs: { name: 'keywords', content: 'llm,gateway,rust,openai,anthropic,api,providers,llmg' },
        },
        {
          tag: 'meta',
          attrs: { name: 'robots', content: 'index, follow' },
        },
      ],
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Introduction', slug: 'getting-started/introduction' },
            { label: 'Installation', slug: 'getting-started/installation' },
            { label: 'Quick Start', slug: 'getting-started/quickstart' },
          ],
        },
        {
          label: 'Gateway',
          items: [
            { label: 'Authentication', slug: 'gateway/authentication' },
            { label: 'Configuration', slug: 'gateway/configuration' },
            { label: 'Docker', slug: 'gateway/docker' },
          ],
        },
        {
          label: 'Providers',
          items: [
            { label: 'Overview', slug: 'providers/overview' },
            { label: 'OpenAI', slug: 'providers/openai' },
            { label: 'Anthropic', slug: 'providers/anthropic' },
            { label: 'Azure OpenAI', slug: 'providers/azure' },
            { label: 'Google Vertex AI', slug: 'providers/vertex-ai' },
            { label: 'AWS Bedrock', slug: 'providers/bedrock' },
            { label: 'Groq', slug: 'providers/groq' },
            { label: 'Mistral', slug: 'providers/mistral' },
            { label: 'Cohere', slug: 'providers/cohere' },
            { label: 'DeepSeek', slug: 'providers/deepseek' },
            { label: 'Z.AI (GLM)', slug: 'providers/z-ai' },
            { label: 'Ollama', slug: 'providers/ollama' },
            { label: 'OpenRouter', slug: 'providers/openrouter' },
            { label: 'All Providers', slug: 'providers/all' },
          ],
        },
        {
          label: 'Library',
          items: [
            { label: 'Usage', slug: 'library/usage' },
            { label: 'Rig Integration', slug: 'library/rig' },
          ],
        },
      ],
    }),
    react(),
    sitemap(),
  ],
});
