"""Yelp Human-vs-AI 리뷰 탐지 파이프라인 모듈.

⚠️ src.train / src.attention 실행 시 TensorFlow가 pandas보다 먼저 import됨
(macOS pyarrow↔TF abseil 데드락 회피) — 각 진입점 상단 참조.
"""
