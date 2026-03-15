import argparse
import base64
import json
import math
import os
import shutil
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Segment:
    id: str
    name: str
    lat: float
    lng: float
    heading: int = 0
    notes: str = ""


@dataclass
class Route:
    id: str
    title: str
    distance_m: int
    eta_min: int
    segments: List[Segment] = field(default_factory=list)


class GoogleDirectionsMapAPI:
    """Google Routes API (Directions v2) adapter."""

    ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"

    def __init__(
        self,
        api_key: str,
        origin: str,
        destination: str,
        language: str = "ko",
        region: str = "kr",
        mode: str = "walking",
        blocked_radius_m: float = 25.0,
    ) -> None:
        self.api_key = api_key
        self.origin = origin
        self.destination = destination
        self.language = language
        self.region = region
        self.mode = mode
        self.blocked_radius_m = blocked_radius_m

    @staticmethod
    def _strip_html(text: str) -> str:
        out = text
        replacements = {
            '<div style="font-size:0.9em">': " ",
            "</div>": " ",
            "<b>": "",
            "</b>": "",
            "<wbr/>": "",
        }
        for src, dst in replacements.items():
            out = out.replace(src, dst)
        return " ".join(out.split())

    @staticmethod
    def _parse_duration_seconds(duration_text: str) -> int:
        # Routes API returns duration as a protobuf string, e.g. "742s".
        if not duration_text:
            return 0
        if duration_text.endswith("s"):
            duration_text = duration_text[:-1]
        try:
            return int(float(duration_text))
        except ValueError:
            return 0

    def _travel_mode_for_routes_api(self) -> str:
        mapping = {
            "walking": "WALK",
            "bicycling": "BICYCLE",
        }
        return mapping.get(self.mode, "WALK")

    @staticmethod
    def _bearing(
        start_lat: float, start_lng: float, end_lat: float, end_lng: float
    ) -> int:
        lat1 = math.radians(start_lat)
        lat2 = math.radians(end_lat)
        d_lon = math.radians(end_lng - start_lng)
        x = math.sin(d_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (
            math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
        )
        initial = math.degrees(math.atan2(x, y))
        return int((initial + 360) % 360)

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371000.0
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(d_lon / 2) ** 2
        )
        return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _is_near_blocked_point(
        self, segment: Segment, blocked_points: List[Tuple[float, float]]
    ) -> bool:
        for b_lat, b_lng in blocked_points:
            if (
                self._haversine_m(segment.lat, segment.lng, b_lat, b_lng)
                <= self.blocked_radius_m
            ):
                return True
        return False

    def get_routes(
        self, blocked_points: Optional[List[Tuple[float, float]]] = None
    ) -> List[Route]:
        blocked_points = blocked_points or []
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": (
                "routes.duration,routes.distanceMeters,routes.description,"
                "routes.legs.steps.startLocation,routes.legs.steps.endLocation,"
                "routes.legs.steps.navigationInstruction.instructions"
            ),
        }
        body = {
            "origin": {"address": self.origin},
            "destination": {"address": self.destination},
            "travelMode": self._travel_mode_for_routes_api(),
            "computeAlternativeRoutes": True,
            "languageCode": self.language,
            "regionCode": self.region.upper(),
            "units": "METRIC",
        }

        resp = requests.post(self.ROUTES_URL, headers=headers, json=body, timeout=30)
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        if resp.status_code >= 400:
            error = payload.get("error", {})
            code = error.get("status", f"HTTP_{resp.status_code}")
            msg = error.get("message", resp.text[:300])
            raise RuntimeError(f"Google Routes 오류: {code} - {msg}")
        if "error" in payload:
            error = payload.get("error", {})
            code = error.get("status", "UNKNOWN")
            msg = error.get("message", "Routes API 호출 실패")
            raise RuntimeError(f"Google Routes 오류: {code} - {msg}")

        routes: List[Route] = []
        for r_idx, r in enumerate(payload.get("routes", []), start=1):
            legs = r.get("legs", [])
            if not legs:
                continue
            leg = legs[0]
            steps = leg.get("steps", [])
            if not steps:
                continue

            segments: List[Segment] = []
            for s_idx, step in enumerate(steps, start=1):
                start_loc = step.get("startLocation", {}).get("latLng", {})
                end_loc = step.get("endLocation", {}).get("latLng", {})
                lat = float(start_loc.get("lat", 0.0))
                lng = float(start_loc.get("lng", 0.0))
                end_lat = float(end_loc.get("lat", lat))
                end_lng = float(end_loc.get("lng", lng))
                heading = self._bearing(lat, lng, end_lat, end_lng)
                nav = step.get("navigationInstruction", {})
                name = self._strip_html(nav.get("instructions", f"step_{s_idx}"))

                segments.append(
                    Segment(
                        id=f"R{r_idx}_S{s_idx}",
                        name=name[:80] if name else f"step_{s_idx}",
                        lat=lat,
                        lng=lng,
                        heading=heading,
                        notes="directions_step",
                    )
                )

            if not segments:
                continue

            if blocked_points and any(
                self._is_near_blocked_point(seg, blocked_points) for seg in segments
            ):
                continue

            summary = r.get("description") or "Google Routes 경로"
            distance_m = int(r.get("distanceMeters", 0))
            duration_s = self._parse_duration_seconds(r.get("duration", "0s"))
            routes.append(
                Route(
                    id=f"R{r_idx}",
                    title=summary,
                    distance_m=distance_m,
                    eta_min=max(1, round(duration_s / 60)),
                    segments=segments,
                )
            )

        return sorted(routes, key=lambda x: (x.distance_m, x.eta_min))


def street_view_url(segment: Segment, api_key: str, size: str = "960x540") -> str:
    return (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size={size}&location={segment.lat},{segment.lng}&heading={segment.heading}&pitch=0&fov=95&key={api_key}"
    )


def ensure_image_from_url(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)
    return output_path


def local_image_to_data_url(path: Path) -> str:
    ext = path.suffix.lower().replace(".", "")
    mime = "jpeg" if ext in {"jpg", "jpeg"} else "png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


def extract_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def analyze_with_vlm(client: OpenAI, model: str, image_input: str) -> Dict[str, Any]:
    image_part = {"type": "input_image", "image_url": image_input}

    prompt = textwrap.dedent(
        """
        You are a barrier-free pedestrian safety assessment system.
        Analyze the image from the perspective of wheelchair users, stroller users, and mobility-impaired pedestrians.

        Return ONLY valid JSON in the exact schema below.
        {
          "passable": true_or_false,
          "risk_score": 0_to_100,
          "obstacles": [
            {
              "type": "high_curb|steep_slope|stairs|narrow_sidewalk|construction|blocked_path|surface_damage|unknown",
              "severity": "low|medium|high",
              "confidence": 0_to_1,
              "evidence": "short evidence sentence"
            }
          ],
          "summary": "one-line summary",
          "reroute_needed": true_or_false
        }

        Rules:
        - Classify high curb, stairs, construction barriers, insufficient sidewalk width, and steep slopes as high severity.
        - If uncertain, add an "unknown" obstacle item.
        - When ambiguous, be conservative for safety.
        """
    ).strip()

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    image_part,
                ],
            }
        ],
        temperature=0,
    )

    output_text = response.output_text or "{}"
    return extract_json_block(output_text)


def print_segment_result(
    segment: Segment, local_image_path: Path, result: Dict[str, Any]
) -> None:
    obstacles = result.get("obstacles", [])
    obstacle_text = (
        ", ".join(o.get("type", "unknown") for o in obstacles) if obstacles else "없음"
    )
    print(f"  - 세그먼트: {segment.id} ({segment.name})")
    print(f"    이미지: {local_image_path}")
    print(
        f"    판정: passable={result.get('passable')} risk={result.get('risk_score')} reroute={result.get('reroute_needed')}"
    )
    print(f"    장애물: {obstacle_text}")
    print(f"    요약: {result.get('summary', '')}")


def write_report(
    run_dir: Path, attempts: List[Dict[str, Any]], selected: Optional[Dict[str, Any]]
) -> Path:
    lines: List[str] = ["# Barrier-Free Navigation Report", ""]
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    for attempt in attempts:
        route = attempt["route"]
        lines.append(f"## Attempt {attempt['loop']} - Route {route.id} ({route.title})")
        lines.append(f"- distance: {route.distance_m}m")
        lines.append(f"- eta: {route.eta_min}min")
        lines.append("")

        for seg_eval in attempt["segment_results"]:
            segment: Segment = seg_eval["segment"]
            result = seg_eval["result"]
            rel_path = seg_eval["image_path"].relative_to(run_dir)
            lines.append(f"### {segment.id} {segment.name}")
            lines.append(f"![{segment.id}](./{rel_path.as_posix()})")
            lines.append("```json")
            lines.append(json.dumps(result, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")

        if attempt["blocked_segments"]:
            lines.append(
                f"- blocked_segments: {', '.join(attempt['blocked_segments'])}"
            )
        else:
            lines.append("- blocked_segments: none")
        lines.append("")

    lines.append("## Final Selection")
    if selected:
        route = selected["route"]
        lines.append(f"- route_id: {route.id}")
        lines.append(f"- title: {route.title}")
        lines.append(f"- distance: {route.distance_m}m")
        lines.append(f"- eta: {route.eta_min}min")
    else:
        lines.append("- no safe route found in max loop")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def configure_console_utf8() -> None:
    if os.name == "nt":
        os.system("chcp 65001 >NUL")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def run(args: argparse.Namespace) -> None:
    configure_console_utf8()

    if not args.openai_api_key and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY를 설정하거나 --openai-api-key를 전달하세요.")

    if not args.google_maps_api_key and not os.getenv("GOOGLE_MAPS_API_KEY"):
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY를 설정하거나 --google-maps-api-key를 전달하세요."
        )

    openai_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    google_key = args.google_maps_api_key or os.getenv("GOOGLE_MAPS_API_KEY")

    client = OpenAI(api_key=openai_key)
    map_api = GoogleDirectionsMapAPI(
        api_key=google_key,
        origin=args.start,
        destination=args.end,
        language=args.language,
        region=args.region,
        mode=args.travel_mode,
        blocked_radius_m=args.blocked_radius_m,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"run_{ts}"
    img_dir = run_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    blocked_points: List[Tuple[float, float]] = []
    attempts: List[Dict[str, Any]] = []
    selected: Optional[Dict[str, Any]] = None

    print("\n=== LM 기반 배리어프리 경로 탐색 시작 ===")
    print(f"출발지: {args.start}")
    print(f"도착지: {args.end}")
    print(f"모델: {args.model}")
    print(f"이미지 모드: {args.image_mode}")
    print(f"지도 경로 모드: Google Directions ({args.travel_mode})")

    for loop_idx in range(1, args.max_loops + 1):
        candidates = map_api.get_routes(blocked_points=blocked_points)
        if not candidates:
            print("\n더 이상 후보 경로가 없습니다.")
            break

        route = candidates[0]
        print(
            f"\n[Loop {loop_idx}] 지도 API 1차 경로: {route.id} ({route.title}, {route.distance_m}m)"
        )

        segment_results: List[Dict[str, Any]] = []
        blocked_in_this_route: List[str] = []

        for idx, segment in enumerate(route.segments, start=1):
            local_path = img_dir / f"loop{loop_idx}_{segment.id}.jpg"

            if args.image_mode == "streetview":
                sv_url = street_view_url(segment, google_key)
                ensure_image_from_url(sv_url, local_path)
                vlm_image_input = sv_url
            else:
                if not args.local_images:
                    raise RuntimeError("local 모드에서는 --local-images가 필요합니다.")
                src = Path(args.local_images[(idx - 1) % len(args.local_images)])
                if not src.exists():
                    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {src}")
                shutil.copy2(src, local_path)
                vlm_image_input = local_image_to_data_url(src)

            result = analyze_with_vlm(client, args.model, vlm_image_input)
            print_segment_result(segment, local_path, result)

            segment_results.append(
                {
                    "segment": segment,
                    "image_path": local_path,
                    "result": result,
                }
            )

            should_reroute = bool(result.get("reroute_needed")) or not bool(
                result.get("passable", True)
            )
            if should_reroute:
                blocked_in_this_route.append(segment.id)
                blocked_points.append((segment.lat, segment.lng))

        attempts.append(
            {
                "loop": loop_idx,
                "route": route,
                "segment_results": segment_results,
                "blocked_segments": blocked_in_this_route,
            }
        )

        if blocked_in_this_route:
            print(
                f"\n장애물 감지 -> 재탐색: {', '.join(blocked_in_this_route)} 세그먼트 회피"
            )
            continue

        selected = attempts[-1]
        print("\n안전한 경로 확정 완료")
        break

    report_path = write_report(run_dir, attempts, selected)
    print(f"\n리포트 저장: {report_path.resolve()}")

    if selected:
        route = selected["route"]
        print(
            f"선택 경로: {route.id} | 거리 {route.distance_m}m | 예상 {route.eta_min}분"
        )
    else:
        print("결과: 최대 루프 내에 안전 경로를 찾지 못했습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LM 기반 배리어프리 내비게이션 에이전트 데모"
    )
    parser.add_argument("--start", default="서울시청")
    parser.add_argument("--end", default="경복궁역")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--max-loops", type=int, default=4)
    parser.add_argument(
        "--image-mode", choices=["streetview", "local"], default="streetview"
    )
    parser.add_argument(
        "--local-images", nargs="*", help="local 모드에서 사용할 이미지 경로들"
    )
    parser.add_argument(
        "--travel-mode", choices=["walking", "bicycling"], default="walking"
    )
    parser.add_argument("--language", default="ko")
    parser.add_argument("--region", default="kr")
    parser.add_argument("--blocked-radius-m", type=float, default=25.0)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--google-maps-api-key", default=None)

    run(parser.parse_args())
