import type { ModelInfo } from "../types";

interface Props {
  model: ModelInfo | null;
}

export function ModelPanel({ model }: Props) {
  if (!model) return null;
  const card = model.card;

  return (
    <section className="card model-panel">
      <h2 className="card-title">Detector</h2>
      {!card ? (
        <p className="note">
          Running <strong>{model.model}</strong> ({model.backend === "fallback" ? "colour-heuristic fallback" : "trained"}). Install the
          trained model with <code>python scripts/download_model.py</code>.
        </p>
      ) : (
        <>
          <p className="model-name">
            {card.architecture.split(",")[0]}
            {card.parameters !== null && ` · ${(card.parameters / 1e6).toFixed(2)} M parameters`}
            {card.gflops !== null && ` · ${card.gflops} GFLOPs`}
          </p>
          <div className="tiles tiles-4">
            <div className="tile">
              <span className="tile-label">mAP50</span>
              <span className="tile-value">{card.metrics.mAP50.toFixed(3)}</span>
            </div>
            <div className="tile">
              <span className="tile-label">mAP50-95</span>
              <span className="tile-value">{card.metrics.mAP50_95.toFixed(3)}</span>
            </div>
            <div className="tile">
              <span className="tile-label">Precision</span>
              <span className="tile-value">{card.metrics.precision.toFixed(3)}</span>
            </div>
            <div className="tile">
              <span className="tile-label">Recall</span>
              <span className="tile-value">{card.metrics.recall.toFixed(3)}</span>
            </div>
          </div>
          <dl className="facts">
            <dt>Data</dt>
            <dd>
              <a href={card.dataset.url} target="_blank" rel="noreferrer">
                PDD – Apple – Scab
              </a>{" "}
              ({card.dataset.license}): {card.dataset.train_images} train / {card.dataset.val_images} val images, PlantVillage lab photos
            </dd>
            <dt>Training</dt>
            <dd>
              {card.training.epochs_completed} epochs on {card.training.hardware.replace("Google Colab, ", "Colab ")}, best at epoch{" "}
              {card.training.best_epoch}
            </dd>
            <dt>Export</dt>
            <dd>
              {card.export.format.split(" (")[0]}, {(card.export.size_bytes / 1e6).toFixed(1)} MB ·{" "}
              <a href={`https://github.com/peerzadarabet1807/frostguard-scab/releases/tag/${card.export.release_tag}`} target="_blank" rel="noreferrer">
                release {card.export.release_tag}
              </a>
            </dd>
            <dt>Limits</dt>
            <dd>{card.limitations[0]}</dd>
          </dl>
        </>
      )}
    </section>
  );
}
