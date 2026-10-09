import { z } from "zod";

// 1. Summary Schema
export const SummarySchema = z.object({
  dataset_name: z.string(),
  bronze_row_count: z.number(),
  silver_row_count: z.number(),
  gold_row_count: z.number(),
  gold_eval_count: z.number().default(1000),
  class_count: z.number(),
  classes: z.array(z.string()),
  split_counts: z.object({
    train: z.number(),
    val: z.number(),
    test: z.number(),
  }),
  split_dates: z.object({
    train_start: z.string(),
    train_end: z.string(),
    val_start: z.string(),
    val_end: z.string(),
    test_start: z.string(),
    test_end: z.string(),
  }),
  models_compared: z.number(),
  cost_disclaimer: z.string(),
  updated_at: z.string(),
});
export type SummaryData = z.infer<typeof SummarySchema>;

// 2. Pipeline Schema
export const PipelineStageSchema = z.object({
  id: z.string(),
  name: z.string(),
  tool: z.string(),
  count: z.number(),
  description: z.string(),
  status: z.enum(["completed", "passed", "in_progress", "pending"]).default("completed"),
});
export const PipelineSchema = z.object({
  stages: z.array(PipelineStageSchema),
});
export type PipelineData = z.infer<typeof PipelineSchema>;
export type PipelineStage = z.infer<typeof PipelineStageSchema>;

// 3. Models Schema
export const ModelMetricSchema = z.object({
  id: z.string(),
  name: z.string(),
  macro_f1: z.number().nullable(),
  ci_low: z.number().nullable(),
  ci_high: z.number().nullable(),
  accuracy: z.number().nullable(),
  cost_per_1k: z.number().optional().nullable(),
  hypothetical_cost_per_1k: z.number().nullable(),
  cost_label: z.string().default("hypothetical, at reference pricing"),
  latency_p50_ms: z.number().nullable(),
  latency_p95_ms: z.number().nullable(),
  ece: z.number().nullable(),
  n_samples: z.number().nullable(),
  schema_valid_rate: z.number().nullable().optional(),
  is_pareto: z.boolean().default(false),
});
export const ModelsDataSchema = z.object({
  models: z.array(ModelMetricSchema),
  takeaway: z.string().default(""),
  reference_pricing_note: z.string().default("Hypothetical cost, at reference pricing. Actual expenditure is $0.00."),
});
export type ModelMetric = z.infer<typeof ModelMetricSchema>;
export type ModelsData = z.infer<typeof ModelsDataSchema>;

// 4. Pairwise Tests Schema
export const PairwiseTestSchema = z.object({
  model_a: z.string(),
  model_b: z.string(),
  delta_f1: z.number().nullable(),
  delta_f1_ci_lower: z.number().nullable(),
  delta_f1_ci_upper: z.number().nullable(),
  excludes_zero: z.boolean(),
  mcnemar_stat: z.number().nullable(),
  mcnemar_p_value: z.number().nullable(),
  adjusted_p_value: z.number().nullable().optional(),
  significant_after_holm: z.boolean().default(false),
});
export const PairwiseDataSchema = z.object({
  pairs: z.array(PairwiseTestSchema),
});
export type PairwiseTest = z.infer<typeof PairwiseTestSchema>;
export type PairwiseData = z.infer<typeof PairwiseDataSchema>;

// 5. Calibration Schema
export const CalibrationBinSchema = z.object({
  bin_index: z.number(),
  bin_range: z.tuple([z.number(), z.number()]),
  count: z.number(),
  accuracy: z.number(),
  confidence: z.number(),
});
export const ModelCalibrationSchema = z.object({
  ece: z.number(),
  bins: z.array(CalibrationBinSchema),
});
export const CalibrationDataSchema = z.object({
  models: z.record(z.string(), ModelCalibrationSchema),
});
export type CalibrationData = z.infer<typeof CalibrationDataSchema>;

// 6. Bias Schema
export const BiasProbeSchema = z.object({
  name: z.string(),
  description: z.string(),
  flip_rate: z.number().nullable(),
  flips: z.number(),
  total: z.number(),
  ci_low: z.number().nullable(),
  ci_high: z.number().nullable(),
});
export const BiasExampleSchema = z.object({
  probe: z.string(),
  original_text: z.string(),
  perturbed_text: z.string(),
  original_pred: z.string(),
  perturbed_pred: z.string(),
  flipped: z.boolean(),
});
export const BiasDataSchema = z.object({
  eval_sample_size: z.number(),
  probes: z.record(z.string(), BiasProbeSchema),
  examples: z.array(BiasExampleSchema),
  limitations: z.array(z.string()),
});
export type BiasData = z.infer<typeof BiasDataSchema>;

// 7. Volume Schema (sql/04_trends_window.sql)
export const ProductMonthlyTrendSchema = z.object({
  month: z.string(),
  monthly_volume: z.number(),
  share_pct: z.number(),
  prev_month_volume: z.number(),
  rolling_3m_avg: z.number(),
  mom_growth_pct: z.number(),
  monthly_rank: z.number(),
});
export const VolumeSummaryItemSchema = z.object({
  product: z.string(),
  total_volume: z.number(),
  avg_share_pct: z.number(),
  latest_month: z.string(),
  latest_volume: z.number(),
  latest_mom_growth: z.number(),
  latest_rolling_3m_avg: z.number(),
});
export const OverallMonthlyItemSchema = z.object({
  month: z.string(),
  total_volume: z.number(),
});
export const VolumeDataSchema = z.object({
  time_range: z.string(),
  total_complaints: z.number(),
  products: z.array(z.string()),
  by_product: z.record(z.string(), z.array(ProductMonthlyTrendSchema)),
  overall_monthly: z.array(OverallMonthlyItemSchema),
  summary_table: z.array(VolumeSummaryItemSchema),
});
export type ProductMonthlyTrend = z.infer<typeof ProductMonthlyTrendSchema>;
export type VolumeSummaryItem = z.infer<typeof VolumeSummaryItemSchema>;
export type VolumeData = z.infer<typeof VolumeDataSchema>;

// 8. Examples Schema
export const ModelPredictionSchema = z.object({
  prediction: z.string(),
  confidence: z.number(),
  correct: z.boolean(),
});
export const LLMExtractionSchema = z.object({
  issue_category: z.string(),
  root_cause: z.string(),
  urgency: z.string(),
  sentiment: z.string(),
  requested_resolution: z.string(),
  evidence_quote: z.string(),
});
export const ComplaintExampleSchema = z.object({
  complaint_id: z.string(),
  narrative: z.string(),
  true_label: z.string(),
  word_count: z.number(),
  model_predictions: z.record(z.string(), ModelPredictionSchema),
  llm_extraction: LLMExtractionSchema.nullable().optional(),
});
export const ExamplesDataSchema = z.object({
  examples: z.array(ComplaintExampleSchema),
});
export type ComplaintExample = z.infer<typeof ComplaintExampleSchema>;
export type ExamplesData = z.infer<typeof ExamplesDataSchema>;

// 9. Gate Schema
export const GateCheckSchema = z.object({
  check_name: z.string(),
  status: z.enum(["PASS", "FAIL", "NOT RUN"]),
  details: z.string().optional(),
  observed_value: z.any().optional(),
  threshold_value: z.any().optional(),
});
export const GateDataSchema = z.object({
  passed: z.boolean(),
  checks: z.array(GateCheckSchema),
  candidate_model: z.string(),
  baseline_model: z.string(),
  updated_at: z.string(),
});
export type GateData = z.infer<typeof GateDataSchema>;
