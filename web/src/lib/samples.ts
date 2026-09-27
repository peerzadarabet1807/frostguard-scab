// Held-out PlantVillage leaves (CC BY-SA 3.0), bundled from ../data/samples at build time.
const files = import.meta.glob("../../../data/samples/*.jpg", {
  eager: true,
  query: "?url",
  import: "default",
}) as Record<string, string>;

export interface SampleLeaf {
  id: string;
  label: string;
  url: string;
  healthy: boolean;
}

export const SAMPLE_LEAVES: SampleLeaf[] = Object.entries(files)
  .map(([path, url]) => {
    const id = path.split("/").pop()!.replace(".jpg", "");
    const healthy = id.startsWith("healthy");
    const n = id.split("_").pop();
    return { id, url, healthy, label: healthy ? "Healthy leaf" : `Scab leaf ${n}` };
  })
  .sort((a, b) => Number(a.healthy) - Number(b.healthy) || a.id.localeCompare(b.id));

export const SAMPLE_ATTRIBUTION =
  "Sample photos: PlantVillage dataset (Hughes & Salathé 2015), CC BY-SA 3.0; held out from training.";
