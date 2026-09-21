"""난수 시드 고정 유틸.

`random` / `numpy` / `torch`(+cuDNN deterministic) / `tensorflow` 의 시드를
한 번에 고정한다.

사용 예:
    from src.seed import set_seed
    set_seed()          # 기본값 42 (= src/config.py 의 SEED)
    set_seed(43)        # 5-Run 반복 실험(seed 42~46)

기존 코드와의 관계 — **동작을 바꾸지 않는다**
---------------------------------------------
* `src/config.py` 의 `SEED = 42` 가 이 모듈의 기본값이다.
* `src/train.py` 에는 이미 `set_seed()` 가 있고 `random` / `numpy` /
  `tf.random` 세 가지를 고정한다. 본 모듈은 **그 함수를 대체하거나 수정하지
  않는다.** `src/train.py` 의 동작은 종전 그대로다.
* 본 모듈은 노트북(①② PyTorch 비교 모델)처럼 `src/train.py` 를 거치지 않는
  경로에서 쓰라고 만든 것이다. PyTorch 쪽 시드와 cuDNN determinism 까지
  한 번에 덮는다.

★ TensorFlow import 순서 주의
------------------------------
`src/config.py` 의 경고대로, 이 저장소에서 TensorFlow 는 pandas/pyarrow 보다
**먼저** import 해야 한다(abseil 심볼 충돌 → 첫 fit() 에서 영구 데드락).

그래서 이 모듈은 TensorFlow 를 **직접 import 하지 않는다.** 이미 import 되어
있을 때만(`sys.modules` 확인) 시드를 건다. 즉 `from src.seed import set_seed`
가 import 순서를 망가뜨리는 일은 없다.

TF 시드까지 확실히 걸려면 **TF 를 먼저 import 한 뒤** 호출한다:

    import tensorflow as tf      # ← 반드시 pandas 보다 먼저
    import pandas as pd
    from src.seed import set_seed
    set_seed()                   # 이제 tf.random 까지 고정된다

무엇이 실제로 고정됐는지는 반환값(dict)으로 확인할 수 있다.
"""

from __future__ import annotations

import os
import random
import sys

try:  # 패키지로 import 될 때
    from .config import SEED as DEFAULT_SEED
except ImportError:  # 스크립트로 직접 실행될 때
    DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> dict:
    """시드를 한 번에 고정하고, 실제로 고정한 항목을 dict 로 돌려준다.

    Args:
        seed: 고정할 시드. 기본값은 `src/config.py` 의 `SEED`(42).
        deterministic: True 면 cuDNN 의 비결정적 커널 선택을 끈다.
            재현성은 올라가지만 GPU 학습이 느려질 수 있다.

    Returns:
        {"random": True, "numpy": True, "torch": bool, "tensorflow": bool}
    """
    done = {"random": False, "numpy": False, "torch": False, "tensorflow": False}

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    done["random"] = True

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
        done["numpy"] = True

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        done["torch"] = True

    # ★ TF 는 직접 import 하지 않는다 (위 'import 순서 주의' 참고).
    #   이미 로드된 경우에만 시드를 건다.
    tf = sys.modules.get("tensorflow")
    if tf is not None:
        tf.random.set_seed(seed)
        done["tensorflow"] = True

    return done


if __name__ == "__main__":
    print(f"seed={DEFAULT_SEED} ->", set_seed())
