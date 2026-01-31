// CRITICAL
"use client";

import { useEffect, useMemo, useState } from "react";
import { RefreshCw, Sparkles, Image as ImageIcon, X } from "lucide-react";
import api from "@/lib/api";
import type { RecipeWithStatus } from "@/lib/types";

interface ImageResult {
  id: string;
  dataUrl: string;
}

const DEFAULT_WIDTH = 1024;
const DEFAULT_HEIGHT = 1024;
const DEFAULT_STEPS = 20;
const DEFAULT_CFG = 2.5;

export default function ImageLab() {
  const [recipes, setRecipes] = useState<RecipeWithStatus[]>([]);
  const [loadingRecipes, setLoadingRecipes] = useState(false);
  const [selectedRecipeId, setSelectedRecipeId] = useState<string>("");
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  const [height, setHeight] = useState(DEFAULT_HEIGHT);
  const [steps, setSteps] = useState(DEFAULT_STEPS);
  const [cfgScale, setCfgScale] = useState(DEFAULT_CFG);
  const [sampler, setSampler] = useState("euler");
  const [seed, setSeed] = useState<number | "">(-1);
  const [batchCount, setBatchCount] = useState(1);
  const [images, setImages] = useState<ImageResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const sdRecipes = useMemo(
    () => recipes.filter((recipe) => recipe.backend === "sdcpp"),
    [recipes],
  );

  const selectedRecipe = useMemo(
    () => sdRecipes.find((recipe) => recipe.id === selectedRecipeId) || null,
    [sdRecipes, selectedRecipeId],
  );

  const loadRecipes = async () => {
    setLoadingRecipes(true);
    try {
      const result = await api.getRecipes();
      setRecipes(result.recipes || []);
    } catch (err) {
      setError((err as Error).message || "Failed to load recipes");
    } finally {
      setLoadingRecipes(false);
    }
  };

  useEffect(() => {
    void loadRecipes();
  }, []);

  useEffect(() => {
    if (!selectedRecipeId && sdRecipes.length > 0) {
      setSelectedRecipeId(sdRecipes[0]?.id || "");
    }
  }, [selectedRecipeId, sdRecipes]);

  const handleGenerate = async () => {
    setError(null);
    if (!prompt.trim()) {
      setError("Prompt is required");
      return;
    }
    if (!selectedRecipeId) {
      setError("Select an sd.cpp recipe first");
      return;
    }

    setIsGenerating(true);
    try {
      const payload: Record<string, unknown> = {
        model: selectedRecipeId,
        prompt,
        negative_prompt: negativePrompt || undefined,
        width,
        height,
        steps,
        cfg_scale: cfgScale,
        sampling_method: sampler,
        seed: seed === "" ? undefined : seed,
        n: batchCount,
      };
      const response = await fetch("/api/proxy/v1/images/generations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Failed to generate image");
      }
      const data = await response.json();
      const results = Array.isArray(data?.data) ? data.data : [];
      const nextImages = results
        .filter((item: { b64_json?: string }) => item?.b64_json)
        .map((item: { b64_json: string }, index: number) => ({
          id: `${Date.now()}-${index}`,
          dataUrl: `data:image/png;base64,${item.b64_json}`,
        }));
      setImages(nextImages);
    } catch (err) {
      setError((err as Error).message || "Failed to generate image");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0d0d0d] text-[#e8e6e3]">
      <div className="flex items-center justify-between border-b border-[#1f1f1f] px-6 py-4">
        <div>
          <h1 className="text-xl font-semibold">Image Lab</h1>
          <p className="text-xs text-[#9a9088] mt-1">
            Generate images with sd.cpp recipes (Qwen-Image GGUF).
          </p>
        </div>
        <button
          onClick={loadRecipes}
          disabled={loadingRecipes}
          className="p-2 hover:bg-[#1f1f1f] rounded-lg transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${loadingRecipes ? "animate-spin" : ""}`} />
        </button>
      </div>

      <div className="flex-1 overflow-auto px-6 py-6">
        <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
          <div className="space-y-4">
            <div className="bg-[#1b1b1b] border border-[#363432] rounded-lg p-4 space-y-4">
              <div>
                <label className="block text-sm text-[#9a9088] mb-2">sd.cpp Recipe</label>
                <select
                  value={selectedRecipeId}
                  onChange={(e) => setSelectedRecipeId(e.target.value)}
                  className="w-full px-3 py-2 bg-[#0d0d0d] border border-[#363432] rounded-lg text-sm focus:outline-none focus:border-[#d97706]"
                >
                  <option value="">Select a recipe...</option>
                  {sdRecipes.map((recipe) => (
                    <option key={recipe.id} value={recipe.id}>
                      {recipe.name || recipe.id}
                    </option>
                  ))}
                </select>
                {sdRecipes.length === 0 && (
                  <p className="text-xs text-[#9a9088] mt-2">
                    No sd.cpp recipes found. Create one in Recipes first.
                  </p>
                )}
              </div>

              <div>
                <label className="block text-sm text-[#9a9088] mb-2">Prompt</label>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={4}
                  className="w-full px-3 py-2 bg-[#0d0d0d] border border-[#363432] rounded-lg text-sm focus:outline-none focus:border-[#d97706]"
                />
              </div>

              <div>
                <label className="block text-sm text-[#9a9088] mb-2">Negative Prompt</label>
                <textarea
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  rows={2}
                  className="w-full px-3 py-2 bg-[#0d0d0d] border border-[#363432] rounded-lg text-sm focus:outline-none focus:border-[#d97706]"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-[#9a9088] mb-1">Width</label>
                  <input
                    type="number"
                    value={width}
                    onChange={(e) => setWidth(Number(e.target.value) || DEFAULT_WIDTH)}
                    className="w-full px-3 py-2 bg-[#0d0d0d] border border-[#363432] rounded-lg text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs text-[#9a9088] mb-1">Height</label>
                  <input
                    type="number"
                    value={height}
                    onChange={(e) => setHeight(Number(e.target.value) || DEFAULT_HEIGHT)}
                    className="w-full px-3 py-2 bg-[#0d0d0d] border border-[#363432] rounded-lg text-sm"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-[#9a9088] mb-1">Steps</label>
                  <input
                    type="number"
                    value={steps}
                    onChange={(e) => setSteps(Number(e.target.value) || DEFAULT_STEPS)}
                    className="w-full px-3 py-2 bg-[#0d0d0d] border border-[#363432] rounded-lg text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs text-[#9a9088] mb-1">CFG</label>
                  <input
                    type="number"
                    step={0.1}
                    value={cfgScale}
                    onChange={(e) => setCfgScale(Number(e.target.value) || DEFAULT_CFG)}
                    className="w-full px-3 py-2 bg-[#0d0d0d] border border-[#363432] rounded-lg text-sm"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-[#9a9088] mb-1">Sampler</label>
                  <input
                    type="text"
                    value={sampler}
                    onChange={(e) => setSampler(e.target.value)}
                    className="w-full px-3 py-2 bg-[#0d0d0d] border border-[#363432] rounded-lg text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs text-[#9a9088] mb-1">Seed</label>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(e.target.value === "" ? "" : Number(e.target.value))}
                    className="w-full px-3 py-2 bg-[#0d0d0d] border border-[#363432] rounded-lg text-sm"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs text-[#9a9088] mb-1">Batch Count</label>
                <input
                  type="number"
                  value={batchCount}
                  onChange={(e) => setBatchCount(Math.max(1, Number(e.target.value) || 1))}
                  className="w-full px-3 py-2 bg-[#0d0d0d] border border-[#363432] rounded-lg text-sm"
                />
              </div>

              {error && (
                <div className="text-xs text-[#f87171] bg-[#1f1f1f] border border-[#3b2a2a] rounded p-2">
                  {error}
                </div>
              )}

              <button
                onClick={handleGenerate}
                disabled={isGenerating || !selectedRecipe}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-[#d97706] hover:bg-[#b45309] text-white rounded-lg transition-colors disabled:opacity-60"
              >
                <Sparkles className="w-4 h-4" />
                {isGenerating ? "Generating..." : "Generate"}
              </button>
            </div>

            {selectedRecipe && (
              <div className="text-xs text-[#9a9088]">
                Active recipe: <span className="text-[#e8e6e3]">{selectedRecipe.name || selectedRecipe.id}</span>
              </div>
            )}
          </div>

          <div className="bg-[#1b1b1b] border border-[#363432] rounded-lg p-4 min-h-[400px]">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 text-sm text-[#9a9088]">
                <ImageIcon className="w-4 h-4" />
                Results
              </div>
              {images.length > 0 && (
                <button
                  onClick={() => setImages([])}
                  className="flex items-center gap-1 text-xs text-[#9a9088] hover:text-[#e8e6e3]"
                >
                  <X className="w-3 h-3" />
                  Clear
                </button>
              )}
            </div>

            {images.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-[320px] text-[#6a6560] text-sm">
                <ImageIcon className="w-8 h-8 mb-3" />
                No images yet. Run a prompt to generate.
              </div>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2">
                {images.map((image) => (
                  <div
                    key={image.id}
                    className="overflow-hidden rounded-lg border border-[#2f2c2a] bg-[#0d0d0d]"
                  >
                    <img src={image.dataUrl} alt="generated" className="w-full h-auto" />
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
