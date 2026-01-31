// CRITICAL
import type { Hono } from "hono";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { AppContext } from "../types/context";
import type { LaunchResult, ProcessInfo, Recipe } from "../types/models";
import { AsyncLock, delay } from "../core/async";
import { badRequest, notFound } from "../core/errors";
import { parseRecipe } from "../stores/recipe-serializer";

const switchLock = new AsyncLock();
const launchCancelControllers = new Map<string, AbortController>();

/**
 * Register lifecycle routes.
 * @param app - Hono app.
 * @param context - App context.
 */
export const registerLifecycleRoutes = (app: Hono, context: AppContext): void => {
  /**
   * Check if a process id exists.
   * @param pid - Process id.
   * @returns True if process exists.
   */
  const pidExists = (pid: number): boolean => {
    try {
      process.kill(pid, 0);
      return true;
    } catch {
      return false;
    }
  };

  /**
   * Read last N characters from a log file.
   * @param path - Log file path.
   * @param limit - Max chars.
   * @returns Log tail string.
   */
  const readLogTail = (path: string, limit: number): string => {
    if (!existsSync(path)) {
      return "";
    }
    try {
      const content = readFileSync(path, "utf-8");
      return content.slice(Math.max(0, content.length - limit));
    } catch {
      return "";
    }
  };

  const serializeRecipeDetail = (recipe: Recipe): Record<string, unknown> => {
    const payload: Record<string, unknown> = {
      ...recipe,
      tp: recipe.tensor_parallel_size,
      pp: recipe.pipeline_parallel_size,
    };
    delete payload["tensor_parallel_size"];
    delete payload["pipeline_parallel_size"];
    return payload;
  };

  /**
   * Check if the running process matches the recipe model identity.
   * @param recipe - Recipe data.
   * @param current - Process info.
   * @returns True if the process already serves this recipe.
   */
  const isSameModel = (recipe: Recipe, current: ProcessInfo): boolean => {
    if (recipe.backend === "sdcpp" && current.backend === "sdcpp") {
      return true;
    }
    if (current.served_model_name && recipe.served_model_name && current.served_model_name === recipe.served_model_name) {
      return true;
    }
    if (current.model_path && recipe.model_path) {
      const normalize = (value: string): string => value.replace(/\/+$/, "");
      const currentPath = normalize(current.model_path);
      const recipePath = normalize(recipe.model_path);
      if (currentPath === recipePath) {
        return true;
      }
      return currentPath.split("/").pop() === recipePath.split("/").pop();
    }
    return false;
  };

  app.get("/recipes", async (ctx) => {
    const recipes = context.stores.recipeStore.list();
    const launchingId = context.launchState.getLaunchingRecipeId();
    const result = await Promise.all(recipes.map(async (recipe) => {
      let status = "stopped";
      if (launchingId === recipe.id) {
        status = "starting";
      }
      const current = await context.processManager.findInferenceProcess(recipe.port ?? context.config.inference_port);
      if (current) {
        if (recipe.backend === "sdcpp" && current.backend === "sdcpp") {
          status = "running";
        } else if (current.served_model_name && recipe.served_model_name === current.served_model_name) {
          status = "running";
        } else if (current.model_path) {
          // Compare normalized paths (exact match)
          const normalize = (p: string): string => p.replace(/\/+$/, "");
          if (normalize(recipe.model_path) === normalize(current.model_path)) {
            status = "running";
          } else if (current.model_path.split("/").pop() === recipe.model_path.split("/").pop()) {
            status = "running";
          }
        }
      }
      return { ...recipe, status };
    }));
    return ctx.json(result);
  });

  app.get("/recipes/:recipeId", async (ctx) => {
    const recipeId = ctx.req.param("recipeId");
    const recipe = context.stores.recipeStore.get(recipeId);
    if (!recipe) {
      throw notFound("Recipe not found");
    }
    return ctx.json(serializeRecipeDetail(recipe));
  });

  app.post("/recipes", async (ctx) => {
    const body = await ctx.req.json();
    try {
      const recipe = parseRecipe(body);
      context.stores.recipeStore.save(recipe);
      return ctx.json({ success: true, id: recipe.id });
    } catch (error) {
      throw badRequest(String(error));
    }
  });

  app.put("/recipes/:recipeId", async (ctx) => {
    const recipeId = ctx.req.param("recipeId");
    const body = await ctx.req.json();
    try {
      const recipe = parseRecipe({ ...body, id: recipeId });
      context.stores.recipeStore.save(recipe);
      return ctx.json({ success: true, id: recipe.id });
    } catch (error) {
      throw badRequest(String(error));
    }
  });

  app.delete("/recipes/:recipeId", async (ctx) => {
    const recipeId = ctx.req.param("recipeId");
    const deleted = context.stores.recipeStore.delete(recipeId);
    if (!deleted) {
      throw notFound("Recipe not found");
    }
    return ctx.json({ success: true });
  });

  app.post("/launch/:recipeId", async (ctx) => {
    const recipeId = ctx.req.param("recipeId");
    const recipe = context.stores.recipeStore.get(recipeId);
    if (!recipe) {
      throw notFound("Recipe not found");
    }
    const targetPort = recipe.port ?? context.config.inference_port;
    const alreadyRunning = await context.processManager.findInferenceProcess(targetPort);
    if (alreadyRunning && isSameModel(recipe, alreadyRunning)) {
      await context.eventManager.publishLaunchProgress(recipeId, "ready", "Model is already running", 1.0);
      return ctx.json({
        success: true,
        pid: alreadyRunning.pid,
        message: "Model is already running",
        log_file: join("/tmp", `vllm_${recipeId}.log`),
      } as LaunchResult);
    }

    const currentLaunching = context.launchState.getLaunchingRecipeId();
    if (currentLaunching && currentLaunching !== recipeId) {
      await context.eventManager.publishLaunchProgress(recipeId, "preempting", `Cancelling ${currentLaunching}...`, 0);
      await context.eventManager.publishLaunchProgress(currentLaunching, "cancelled", `Preempted by ${recipeId}`, 0);
      const cancel = launchCancelControllers.get(currentLaunching);
      if (cancel) {
        cancel.abort();
      }
      await context.processManager.evictModel(true, recipe.port);
      if (recipe.backend === "sdcpp") {
        await context.processManager.evictModel(true, context.config.inference_port);
      }
      await delay(1000);
    }

    const cancelController = new AbortController();
    launchCancelControllers.set(recipeId, cancelController);
    context.launchState.setLaunchingRecipeId(recipeId);

    let releaseLock: (() => void) | null = null;
    try {
      releaseLock = await switchLock.acquireWithTimeout(2000);
      if (!releaseLock) {
        await context.processManager.evictModel(true, recipe.port);
        await delay(1000);
        releaseLock = await switchLock.acquire();
      }

      await context.eventManager.publishLaunchProgress(recipeId, "evicting", "Clearing VRAM...", 0);
      await context.processManager.evictModel(true, recipe.port);
      if (recipe.backend === "sdcpp") {
        await context.processManager.evictModel(true, context.config.inference_port);
      }
      for (let attempt = 0; attempt < 10; attempt += 1) {
        const remaining = await context.processManager.findInferenceProcess(targetPort);
        if (!remaining) {
          break;
        }
        await delay(500);
      }
      await delay(1000);

      if (cancelController.signal.aborted) {
        await context.eventManager.publishLaunchProgress(recipeId, "cancelled", "Preempted by another launch", 0);
        return ctx.json({ success: false, pid: null, message: "Launch cancelled", log_file: null } as LaunchResult);
      }

      await context.eventManager.publishLaunchProgress(recipeId, "launching", `Starting ${recipe.name}...`, 0.25);
      const launch = await context.processManager.launchModel(recipe);
      if (!launch.success) {
        await context.eventManager.publishLaunchProgress(recipeId, "error", launch.message, 0);
        return ctx.json({ success: false, pid: null, message: launch.message, log_file: null } as LaunchResult);
      }

      await context.eventManager.publishLaunchProgress(recipeId, "waiting", "Waiting for model to load...", 0.5);

      const start = Date.now();
      const timeout = 300_000;
      let ready = false;
      let fatalError: string | null = null;
      const healthPort = recipe.port ?? context.config.inference_port;
      const healthUrl = `http://${context.config.inference_host}:${healthPort}/health`;
      const fatalPatterns = [
        // vLLM/PyTorch patterns
        "raise ValueError",
        "raise RuntimeError",
        "CUDA out of memory",
        "OutOfMemoryError",
        "torch.OutOfMemoryError",
        "not enough memory",
        "Cannot allocate",
        "larger than the available KV cache memory",
        "EngineCore failed to start",
        // llama.cpp patterns
        "failed to load model",
        "error loading model",
        "GGML_ASSERT",
        "ggml_cuda_error",
        "not enough VRAM",
        "failed to allocate",
        "model file not found",
        "invalid model file",
      ];

      const logFilePath = join("/tmp", `vllm_${recipeId}.log`);

      while (Date.now() - start < timeout) {
        if (cancelController.signal.aborted) {
          await context.eventManager.publishLaunchProgress(recipeId, "cancelled", "Preempted by another launch", 0);
          if (launch.pid && pidExists(launch.pid)) {
            await context.processManager.killProcess(launch.pid, true);
          }
          return ctx.json({ success: false, pid: null, message: "Launch cancelled", log_file: null } as LaunchResult);
        }

        const logTail = readLogTail(logFilePath, 3000);
        if (logTail) {
          for (const pattern of fatalPatterns) {
            if (logTail.includes(pattern)) {
              const lines = logTail.split("\n");
              const index = lines.findIndex((line) => line.includes(pattern));
              if (index >= 0) {
                fatalError = lines.slice(Math.max(0, index - 1), index + 3).join("\n");
              }
              break;
            }
          }
        }
        if (fatalError) {
          break;
        }

        try {
          const controller = new AbortController();
          const timeoutHandle = setTimeout(() => controller.abort(), 5000);
          const response = await fetch(healthUrl, {
            signal: controller.signal,
            headers: context.config.inference_api_key
              ? { Authorization: `Bearer ${context.config.inference_api_key}` }
              : undefined,
          });
          clearTimeout(timeoutHandle);
          if (response.status === 200) {
            ready = true;
            break;
          }
        } catch {
          ready = false;
        }

        if (launch.pid && !pidExists(launch.pid)) {
          const exitTail = readLogTail(logFilePath, 2000);
          fatalError = exitTail
            ? `Process exited early: ${exitTail.slice(-500)}`
            : "Process exited early";
          break;
        }

        const elapsed = Math.floor((Date.now() - start) / 1000);
        await context.eventManager.publishLaunchProgress(
          recipeId,
          "waiting",
          `Loading model... (${elapsed}s)`,
          0.5 + (elapsed / (timeout / 1000)) * 0.5,
        );
        await delay(2000);
      }

      if (fatalError) {
        if (launch.pid) {
          await context.processManager.killProcess(launch.pid, true);
        }
        await context.eventManager.publishLaunchProgress(recipeId, "error", `Fatal error detected: ${fatalError.slice(0, 100)}`, 0);
        return ctx.json({
          success: false,
          pid: null,
          message: `Fatal error: ${fatalError.slice(0, 300)}`,
          log_file: logFilePath,
        } as LaunchResult);
      }

      if (ready) {
        await context.eventManager.publishLaunchProgress(recipeId, "ready", "Model is ready!", 1.0);
        return ctx.json({
          success: true,
          pid: launch.pid ?? null,
          message: "Model is ready",
          log_file: logFilePath,
        } as LaunchResult);
      }

      if (launch.pid) {
        await context.processManager.killProcess(launch.pid, true);
      }
      const errorTail = readLogTail(logFilePath, 1000);
      await context.eventManager.publishLaunchProgress(recipeId, "error", "Model failed to become ready (timeout)", 0);
      return ctx.json({
        success: false,
        pid: null,
        message: `Model failed to become ready (timeout): ${errorTail.slice(-200)}`,
        log_file: logFilePath,
      } as LaunchResult);
    } finally {
      if (releaseLock) {
        releaseLock();
      }
      if (context.launchState.getLaunchingRecipeId() === recipeId) {
        context.launchState.setLaunchingRecipeId(null);
      }
      const controller = launchCancelControllers.get(recipeId);
      if (controller === cancelController) {
        launchCancelControllers.delete(recipeId);
      }
    }
  });

  app.post("/launch/:recipeId/cancel", async (ctx) => {
    const recipeId = ctx.req.param("recipeId");
    const recipe = context.stores.recipeStore.get(recipeId);
    const port = recipe?.port ?? context.config.inference_port;
    const cancel = launchCancelControllers.get(recipeId);
    if (!cancel) {
      const current = context.launchState.getLaunchingRecipeId();
      if (current !== recipeId) {
        throw notFound(`No launch in progress for ${recipeId}`);
      }
      await context.processManager.evictModel(true, port);
      return ctx.json({ success: true, message: "Launch aborted via eviction" });
    }
    cancel.abort();
    await context.processManager.evictModel(true, port);
    return ctx.json({ success: true, message: `Launch of ${recipeId} cancelled` });
  });

  app.post("/evict", async (ctx) => {
    const release = await switchLock.acquire();
    try {
      const portParam = ctx.req.query("port");
      const port = portParam ? Number(portParam) : undefined;
      const pid = await context.processManager.evictModel(Boolean(ctx.req.query("force")), port);
      return ctx.json({ success: true, evicted_pid: pid });
    } finally {
      release();
    }
  });

  app.get("/wait-ready", async (ctx) => {
    const timeout = Number(ctx.req.query("timeout") ?? 300);
    const start = Date.now();
    while (Date.now() - start < timeout * 1000) {
      try {
        const controller = new AbortController();
        const timeoutHandle = setTimeout(() => controller.abort(), 5000);
        const response = await fetch(`http://${context.config.inference_host}:${context.config.inference_port}/health`, {
          signal: controller.signal,
        });
        clearTimeout(timeoutHandle);
        if (response.status === 200) {
          return ctx.json({ ready: true, elapsed: Math.floor((Date.now() - start) / 1000) });
        }
      } catch {
        await delay(2000);
      }
      await delay(2000);
    }
    return ctx.json({ ready: false, elapsed: timeout, error: "Timeout waiting for backend" });
  });
};
