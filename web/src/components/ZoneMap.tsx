import "leaflet/dist/leaflet.css";
import { CircleMarker, MapContainer, TileLayer, Tooltip } from "react-leaflet";
import type { LatLngBoundsExpression } from "leaflet";
import type { ChartPalette, ThemeName } from "../lib/theme";
import type { Zone } from "../types";

interface Props {
  zones: Zone[];
  selected: string;
  onSelect: (key: string) => void;
  theme: ThemeName;
  palette: ChartPalette;
}

// Keyless OpenStreetMap tiles; dark mode re-tones them with a CSS filter (see .zone-map in styles.css).
const TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

// Sopore and Baramulla are ~15 km apart, so their labels point away from each other.
const LABEL_DIRECTION: Record<string, "top" | "bottom" | "left" | "right"> = {
  sopore: "right",
  baramulla: "left",
  pulwama: "right",
  shopian: "left",
};

export function ZoneMap({ zones, selected, onSelect, theme, palette }: Props) {
  const bounds: LatLngBoundsExpression = zones.map((z) => [z.latitude, z.longitude] as [number, number]);

  return (
    <MapContainer
      className={`zone-map zone-map-${theme}`}
      bounds={bounds}
      boundsOptions={{ padding: [36, 36] }}
      scrollWheelZoom={false}
      aria-label="Map of Kashmir orchard zones; click a zone to select it"
    >
      <TileLayer
        url={TILE_URL}
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        maxZoom={18}
      />
      {zones.map((z) => {
        const active = z.key === selected;
        const direction = LABEL_DIRECTION[z.key] ?? "top";
        const offset: [number, number] = direction === "left" ? [-10, 0] : direction === "right" ? [10, 0] : [0, -10];
        return (
          <CircleMarker
            key={`${z.key}-${active}-${theme}`}
            center={[z.latitude, z.longitude]}
            radius={active ? 10 : 7}
            pathOptions={{
              color: theme === "dark" ? "#1a1a19" : "#ffffff",
              weight: 2,
              fillColor: active ? palette.zoneSelected : palette.zone,
              fillOpacity: 1,
            }}
            eventHandlers={{ click: () => onSelect(z.key) }}
          >
            <Tooltip direction={direction} offset={offset} permanent className={active ? "zone-label active" : "zone-label"}>
              {z.name}
            </Tooltip>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}
