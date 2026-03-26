# Cream LoRA Trainer — 설정값 문서

## 노드에서 수정 가능한 설정값

| 설정 | 기본값 | 범위 | 설명 |
|---|---|---|---|
| `dataset_path` | (비어있음) | — | 이미지 + `.txt` 캡션 폴더 경로 |
| `sd_scripts_path` | (비어있음) | — | kohya sd-scripts 설치 경로 |
| `ckpt_name` | — | ComfyUI 체크포인트 목록 | 학습에 사용할 SDXL 체크포인트 |
| `lora_name` | (비어있음) | — | 저장될 로라 파일 이름 |
| `target_steps` | 1000 | 10~50000 | 총 학습 스탭 수 |
| `save_steps` | 100 | 0~10000 | N 스탭마다 중간 로라 저장 (0=최종만) |
| `learning_rate` | 0.0003 | 0.00001~0.1 | U-Net 학습률 |
| `lora_rank` | 32 | 4~128 | LoRA rank (network_dim = network_alpha) |
| `vram_mode` | High VRAM | Low VRAM / High VRAM | VRAM 프리셋 |
| `batch_size` | 1 | 1~8 | 한 스탭에서 동시 처리할 이미지 수 |

## 자동 계산되는 설정값

| 설정 | 값 | 설명 |
|---|---|---|
| `text_encoder_lr` | learning_rate / 10 | Text Encoder 학습률 (High VRAM만 적용) |
| `network_alpha` | = lora_rank | LoRA 적용 강도 (1.0 = 100%) |
| `repeat` | 1 (고정) | sd-scripts 폴더 구조에 의해 결정 |
| 저장 경로 | dataset_path/models/ | 학습된 로라가 저장되는 위치 |

## VRAM 모드

| 모드 | `network_train_unet_only` | `text_encoder_lr` | `cache_text_encoder_outputs` | 권장 VRAM |
|---|---|---|---|---|
| **Low VRAM** | true (U-Net만 학습) | 적용 안 됨 | true (TE 출력 캐싱) | 8GB~ |
| **High VRAM** | false (U-Net + TE 학습) | learning_rate / 10 | false (실시간 계산) | 12GB~ |

## 고정 설정값 (수정 불가)

### 해상도
| 설정 | 값 | 설명 |
|---|---|---|
| `resolution` | 1024×1024 | 모든 모드에서 고정 |

### 옵티마이저

| 설정 | 값 | 설명 |
|---|---|---|
| `optimizer_type` | Adafactor | 메모리 효율적인 Adam 변형 |
| `optimizer_args` | scale_parameter=False, relative_step=False, warmup_init=False | Adafactor 고정 파라미터 |
| `lr_scheduler` | constant_with_warmup | 워밍업 후 일정한 학습률 |
| `lr_warmup_steps` | target_steps × 10% (최대 100) | 워밍업 스텝 수 (동적 계산) |

### 네트워크

| 설정 | 값 | 설명 |
|---|---|---|
| `network_module` | networks.lora | 표준 LoRA |
| `scale_weight_norms` | 1 | 가중치 정규화 |

### 데이터셋

| 설정 | 값 | 설명 |
|---|---|---|
| `enable_bucket` | true | 다양한 종횡비 지원 |
| `bucket_no_upscale` | true | 업스케일 없이 버킷 배치 |
| `bucket_reso_steps` | 32 | 버킷 해상도 단위 |
| `caption_extension` | .txt | 캡션 파일 확장자 |
| `keep_tokens` | 1 | 셔플 시 첫 토큰(트리거 워드) 보존 |
| `cache_latents` | true | 잠재 변수 캐싱 |
| `cache_latents_to_disk` | true | 디스크에 캐싱 (VRAM 절약) |

### 정밀도

| 설정 | 값 | 설명 |
|---|---|---|
| `mixed_precision` | bf16 | 학습 정밀도 |
| `fp8_base` | true | 베이스 모델 fp8 |
| `full_bf16` | true | 전체 bf16 |
| `save_precision` | bf16 | 저장 정밀도 |
| `save_model_as` | safetensors | 저장 형식 |

### 학습 안정성

| 설정 | 값 | 설명 |
|---|---|---|
| `no_half_vae` | true | VAE NaN 에러 방지 |
| `noise_offset` | 0.0357 | 어두운/밝은 이미지 품질 개선 |
| `min_snr_gamma` | 5 | 학습 안정성 개선 |
| `xformers` | true | 메모리 효율적 어텐션 |
| `gradient_checkpointing` | true | VRAM 절약 |

### 기타

| 설정 | 값 | 설명 |
|---|---|---|
| `seed` | 0 | 랜덤 시드 |
| `max_data_loader_n_workers` | 0 | 데이터 로더 워커 수 |
