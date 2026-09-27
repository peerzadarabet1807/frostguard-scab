export function Footer({ apiUrl }: { apiUrl: string }) {
  return (
    <footer className="footer">
      <p>
        <strong>Advisory only.</strong> FrostGuard times spray decisions from weather and photos; follow your extension service's schedule and
        fungicide labels.
      </p>
      <p>
        Weather data by <a href="https://open-meteo.com/">Open-Meteo.com</a> (CC BY 4.0), including ERA5 reanalysis (Copernicus / ECMWF).
        Detector trained on{" "}
        <a href="https://universe.roboflow.com/thesis-okplj/pdd-apple-scab">PDD – Apple – Scab</a> (CC BY 4.0). Sample leaves from PlantVillage
        (CC BY-SA 3.0). Map © OpenStreetMap contributors.
      </p>
      <p>
        <a href="https://github.com/peerzadarabet1807/frostguard-scab">Source on GitHub</a> ·{" "}
        <a href={`${apiUrl}/docs`}>API docs</a> · MIT licensed code; model weights AGPL-3.0.
      </p>
    </footer>
  );
}
