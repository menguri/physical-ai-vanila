import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def print_connected_devices():
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        raise RuntimeError(
            "RealSense 카메라를 찾지 못했습니다. USB 연결, 케이블, 권한, SDK 설치 상태를 확인하세요."
        )

    print("연결된 RealSense 장치:")
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        firmware = dev.get_info(rs.camera_info.firmware_version)

        usb_type = "unknown"
        if dev.supports(rs.camera_info.usb_type_descriptor):
            usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)

        print(f"  [{i}] {name}")
        print(f"      Serial   : {serial}")
        print(f"      Firmware : {firmware}")
        print(f"      USB      : {usb_type}")


def main():
    print_connected_devices()

    pipeline = rs.pipeline()
    config = rs.config()

    # D435 기본 테스트용 해상도/프레임
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # depth 좌표계를 color 좌표계에 맞춤
    align = rs.align(rs.stream.color)

    save_dir = Path("realsense_captures")
    save_dir.mkdir(exist_ok=True)

    try:
        profile = pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"\nDepth scale: {depth_scale} meter/unit")
        print("실행 중...")
        print("  q 또는 ESC : 종료")
        print("  s          : 현재 RGB/Depth 이미지 저장\n")

        frame_count = 0
        prev_time = time.time()
        fps = 0.0

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            height, width = depth_image.shape
            cx, cy = width // 2, height // 2

            # 화면 중앙의 거리값, 단위: meter
            center_distance_m = depth_frame.get_distance(cx, cy)

            # Depth 이미지를 보기 좋게 컬러맵으로 변환
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET,
            )

            # RGB 화면에 중앙점/거리 표시
            cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)
            cv2.drawMarker(
                color_image,
                (cx, cy),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )

            text = f"Center depth: {center_distance_m:.3f} m"
            cv2.putText(
                color_image,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # FPS 계산
            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                frame_count = 0
                prev_time = now

            cv2.putText(
                color_image,
                f"FPS: {fps:.1f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            combined = np.hstack((color_image, depth_colormap))
            cv2.imshow("Intel RealSense D435 - Color | Depth", combined)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            if key == ord("s"):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                color_path = save_dir / f"color_{timestamp}.png"
                depth_path = save_dir / f"depth_raw_{timestamp}.png"
                depth_vis_path = save_dir / f"depth_vis_{timestamp}.png"

                cv2.imwrite(str(color_path), color_image)
                cv2.imwrite(str(depth_path), depth_image)
                cv2.imwrite(str(depth_vis_path), depth_colormap)

                print(f"저장 완료:")
                print(f"  {color_path}")
                print(f"  {depth_path}")
                print(f"  {depth_vis_path}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("카메라 스트리밍 종료")


if __name__ == "__main__":
    main()
