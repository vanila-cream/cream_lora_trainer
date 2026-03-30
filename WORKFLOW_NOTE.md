> 설정값에 대한 설명은 마우스를 올리면 툴팁에 표시됩니다.
> 각 노드의 자세한 설명은 readme.md 파일을 확인해주세요.

### 📋 워크플로우 기본 사용 방법

1. **(필수)** Cream Auto Captioner 설정
- dataset_path : 학습에 사용할 데이터셋(이미지들)이 있는 경로를 설정해줍니다. (예: C:\comfyUI\imgs)
- trigger_word : LoRA에서 쓸 트리거워드를 지정해줍니다. 단부루에서 사용되는 태그를 피하는게 좋습니다. 일반적으로 캐릭터의 이름을 leet(문자→숫자 치환)로 변환해서 사용합니다 (예: claw -> c14w)

2. **(필수)** Cream LoRA Trainer 설정
- sd_scripts_path : 설치한 kohya_ss의 sd_scripts 경로를 입력합니다. (예: `C:\Data\Packages\kohya_ss\sd-scripts`)
    (경로를 공란으로 두면 자동으로 `custom_nodes\comfyui-cream-lora-trainer` 폴더 내에 설치됩니다.)
- ckpt_name : 학습에 사용될 모델을 선택합니다. [Illustrious-XL-v1.0.safetensors](https://civitai.com/models/1232765/illustrious-xl-10)를 추천합니다. 혹은 자신이 주로 사용하는 체크포인트로 지정해도 됩니다.
- lora_name : 저장될 LoRA 파일의 이름을 입력합니다.
- vram_mode : 8GB VRAM 그래픽카드를 사용하면 LOW VRAM 모드를 선택합니다.

3. (선택) 샘플 이미지 생성을 위한 긍정 프롬프트 설정
- 완성된 LoRA를 테스트할 때 사용될 이미지 프롬프트를 적어줍니다. 트리거워드나 공통적으로 사용되는 태그는 Cream Common Tag Extractor에서 자동으로 추가되니 구도, 자세, 배경 등만 적어주셔도 됩니다. 비워둬도 상관없습니다. (예: upper_body, straight-on, looking_at_view, outdoors)

4. **(필수)** Efficient Loader 설정
- ckpt_name : 샘플 이미지 생성에 사용할 체크포인트를 설정해줍니다.
- (선택) empty_latent_width/height : 샘플 이미지의 크기를 설정할 수 있습니다.

### 🚀 실행 및 결과

- 설정 완료 후 `실행` 버튼을 누르면 다음 순서로 자동 진행됩니다:
  1. 캡션 생성 (Cream Auto Captioner)
  2. LoRA 학습 (Cream LoRA Trainer)
  3. 샘플 이미지 생성 및 비교 plot 저장

- 결과물 위치
  - 학습된 LoRA 파일: `dataset_path/models/` 폴더
  - 에러 로그: `dataset_path/models/` 폴더 (txt 형식)

- 샘플 이미지 재생성
  - Cream Auto Captioner와 Cream Lora Trainer 노드의 설정을 변경하지 않고 다시 실행하면, 이전 학습 결과를 재사용하여 샘플 이미지만 재생성할 수 있습니다.
  - 캡션만 생성하고 싶은 경우, Groups Bypasser 노드로 `샘플 이미지 생성` 그룹을 비활성화한 뒤 실행하세요.

### 🔧 문제 해결
- sd-scripts 자동 설치가 실패하는 경우: StabilityMatrix에서 kohya_ss를 설치한 뒤 경로를 수동 지정
- 학습 에러 발생 시: dataset_path/models 폴더의 에러 로그를 확인하세요.
- VRAM 부족: Low VRAM 모드로 변경하세요. (batch_size는 자동으로 1로 고정됩니다.)