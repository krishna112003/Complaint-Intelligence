import {
  BiasData,
  BiasDataSchema,
  CalibrationData,
  CalibrationDataSchema,
  ExamplesData,
  ExamplesDataSchema,
  GateData,
  GateDataSchema,
  ModelsData,
  ModelsDataSchema,
  PairwiseData,
  PairwiseDataSchema,
  PipelineData,
  PipelineSchema,
  SummaryData,
  SummarySchema,
  VolumeData,
  VolumeDataSchema,
} from "./types";

const DATA_BASE_URL = "/data";

async function fetchAndValidate<T>(
  filename: string,
  schema: { parse: (val: unknown) => T }
): Promise<T | null> {
  try {
    const res = await fetch(`${DATA_BASE_URL}/${filename}`, {
      cache: "no-store",
    });
    if (!res.ok) {
      console.warn(`[Data Loader] Failed to load ${filename}: ${res.statusText}`);
      return null;
    }
    const json = await res.json();
    return schema.parse(json);
  } catch (err) {
    console.error(`[Data Loader] Error validating ${filename}:`, err);
    return null;
  }
}

export async function getSummaryData(): Promise<SummaryData | null> {
  return fetchAndValidate("summary.json", SummarySchema);
}

export async function getPipelineData(): Promise<PipelineData | null> {
  return fetchAndValidate("pipeline.json", PipelineSchema);
}

export async function getModelsData(): Promise<ModelsData | null> {
  return fetchAndValidate("models.json", ModelsDataSchema);
}

export async function getPairwiseData(): Promise<PairwiseData | null> {
  return fetchAndValidate("pairwise_tests.json", PairwiseDataSchema);
}

export async function getCalibrationData(): Promise<CalibrationData | null> {
  return fetchAndValidate("calibration.json", CalibrationDataSchema);
}

export async function getBiasData(): Promise<BiasData | null> {
  return fetchAndValidate("bias.json", BiasDataSchema);
}

export async function getVolumeData(): Promise<VolumeData | null> {
  return fetchAndValidate("volume.json", VolumeDataSchema);
}

export async function getExamplesData(): Promise<ExamplesData | null> {
  return fetchAndValidate("examples.json", ExamplesDataSchema);
}

export async function getGateData(): Promise<GateData | null> {
  return fetchAndValidate("gate.json", GateDataSchema);
}
