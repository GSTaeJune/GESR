# Berkeley HardFloat — Verilog 생성 환경 셋업 (Windows)

이 문서는 `third_party/berkeley-hardfloat/` 에 들어갈 Verilog 파일을 만드는 과정을 설명합니다. HardFloat 의 canonical repo (`github.com/ucb-bar/berkeley-hardfloat`) 는 **Chisel (Scala DSL)** 로 작성되어 있어서, Verilog 를 얻으려면 Chisel→Verilog 컴파일이 필요합니다.

## 왜 이 셋업이 필요한가

- HardFloat 는 원래 (~2010년대 초) pure Verilog 였지만, 현재 canonical 버전은 Chisel 로 재작성됨.
- Chisel 코드는 JVM 위에서 도는 Scala 컴파일러 가 처리하고, 그 결과로 Verilog 가 emit 됨.
- 한 번만 셋업하면 모든 module 의 Verilog 를 추출해서 `third_party/` 에 vendor 가능. 이후엔 sbt 가 필요 없음.

## 의존성 (한 번만 설치)

| 도구 | 버전 | 용도 |
|---|---|---|
| Java JDK | 11 (LTS, Chisel 3.5.6 권장) | Scala/sbt 가 도는 JVM |
| sbt | 1.9+ | Scala 빌드 툴, Chisel 컴파일 orchestration |
| Git | (이미 설치됨) | HardFloat repo 클론 |

Disk 용량: 첫 설치 ~600 MB (JDK 200 MB + sbt + Maven 캐시).
첫 빌드: 5~10 분 (의존성 다운로드 포함). 이후 빌드: 30 초 ~ 1 분.

## 1. Java JDK 11 설치

### 옵션 A — Eclipse Temurin (추천, 무료, BSD-friendly)

[Adoptium Temurin 11](https://adoptium.net/temurin/releases/?version=11) 에서 Windows MSI 다운로드 → 설치 마법사 실행 → "Set JAVA_HOME variable" 체크박스 켜기.

### 옵션 B — winget (Windows 10+)

```powershell
winget install --id EclipseAdoptium.Temurin.11.JDK
```

### 검증

새 터미널 열고:
```bash
java -version
```

출력 예:
```
openjdk version "11.0.21" 2023-10-17 LTS
OpenJDK Runtime Environment Temurin-11.0.21+9 (build 11.0.21+9-LTS)
```

`java: command not found` 가 나오면 PATH 설정 안 된 것. 시스템 환경변수에 `JAVA_HOME = C:\Program Files\Eclipse Adoptium\jdk-11.x.x-hotspot` 추가 + `Path` 에 `%JAVA_HOME%\bin` 추가.

## 2. sbt 설치

### 옵션 A — 공식 MSI

[sbt 공식 다운로드](https://www.scala-sbt.org/download.html) → Windows MSI → 설치.

### 옵션 B — winget

```powershell
winget install --id Scala.Sbt
```

### 검증

```bash
sbt --version
```

출력 예:
```
sbt runner version: 1.9.7
```

첫 실행 시 Coursier (의존성 fetcher) 가 다운로드되어 약간 시간 걸림.

## 3. HardFloat 클론

```bash
cd /tmp     # 또는 원하는 작업 디렉토리
git clone --depth 1 https://github.com/ucb-bar/berkeley-hardfloat hardfloat_src
cd hardfloat_src
```

## 4. Verilog Emit driver 작성

HardFloat repo 에는 Verilog 를 직접 emit 해주는 entry point 가 없습니다. 우리가 필요한 module 만 emit 하는 작은 Scala driver 를 추가합니다.

다음 파일을 새로 만드세요:

**파일 위치**: `/tmp/hardfloat_src/hardfloat/src/main/scala/EmitVerilog.scala`

```scala
package hardfloat

import chisel3.stage.ChiselStage

object EmitVerilog extends App {
  // RMW 가 쓰는 module 들 (FP32 = expWidth=8, sigWidth=24)
  // 출력 디렉토리는 args(0). 기본값: ./generated_verilog
  val outDir = if (args.length > 0) args(0) else "generated_verilog"

  // 각 module 의 Verilog 를 별도 파일로 emit
  val opts = Array("--target-dir", outDir)

  ChiselStage.emitVerilog(new INToRecFN(32, 8, 24), opts)
  ChiselStage.emitVerilog(new RecFNFromFN(8, 24),   opts)
  ChiselStage.emitVerilog(new AddRecFN(8, 24),      opts)

  // RecFNFromFN 의 inverse 는 hardfloat 안에 fNFromRecFN 으로 들어있음 (lowercase)
  // ChiselStage 가 instance 만들려면 Module wrapper 가 필요 — pure function 이므로
  // 직접 emit 불가. 대신 RecFNToRecFN 을 사용하거나, addRecFN/INToRecFN 결과의
  // 일부 internal logic 으로 등장. 별도로 처리.
  //
  // 실제로 추출되는 .v 파일: INToRecFN.v, RecFNFromFN.v, AddRecFN.v
  // 의존하는 helper class (rawFloatFromFN, RoundAnyRawFNToRecFN, primitives 등) 는
  // 자동으로 같은 .v 파일 안에 포함됨.

  // fNFromRecFN 은 hardware module 이 아니라 utility function. 위 module 들의
  // emit 결과에 inline 으로 포함됨. 만약 별도로 필요하면 다음 wrapper module 을
  // 통해 강제로 추출:
  ChiselStage.emitVerilog(new chisel3.Module {
    val io = chisel3.IO(new chisel3.Bundle {
      val in  = chisel3.Input(chisel3.UInt((8 + 24 + 1).W))
      val out = chisel3.Output(chisel3.UInt((8 + 24).W))
    })
    io.out := fNFromRecFN(8, 24, io.in)
  }, opts ++ Array("--module-name", "FNFromRecFN_wrapper"))
}
```

## 5. Verilog 컴파일

```bash
cd /tmp/hardfloat_src
sbt "runMain hardfloat.EmitVerilog ./generated_verilog"
```

첫 실행:
- Coursier 가 Chisel 3.5.6 + 의존성 다운로드 (~3분)
- Scala 컴파일 (~1분)
- Verilog emit (~10초)
- 총 ~5~10분

성공하면 `./generated_verilog/` 디렉토리에 `.v` 파일들이 생성됨:
- `INToRecFN.v` (≈ 200 줄)
- `RecFNFromFN.v` (≈ 50 줄)
- `AddRecFN.v` (≈ 400 줄, helper 포함)
- `FNFromRecFN_wrapper.v`

**주의**: Chisel emit 결과의 module 이름은 `INToRecFN_<param hash>` 같은 식으로 unique-suffix 가 붙을 수 있습니다. Verilog 파일 안의 `module` 선언 라인을 확인해서 실제 이름을 RTL/TB 에 맞추세요.

## 6. gemm_sram 프로젝트로 복사

```bash
cd C:/Users/ptj72/Desktop/Desktop/00project/gemm_sram
mkdir -p third_party/berkeley-hardfloat
cp /tmp/hardfloat_src/LICENSE                      third_party/berkeley-hardfloat/
cp /tmp/hardfloat_src/generated_verilog/*.v        third_party/berkeley-hardfloat/
```

그리고 `third_party/berkeley-hardfloat/VENDORING.md` 작성 (Task 1 Step 3 참고).

## 7. 검증 (smoke test)

Task 2 (`bash sim/run_rmw_smoke.sh`) 를 돌려서 elaboration 성공 + round-trip 테스트 통과하는지 확인.

만약 elaboration 에서 `cannot find module 'XYZ'` 가 나오면, Chisel 이 inline 시키지 못한 helper module 이 있는 것. 그 module 의 Scala 파일을 찾아 (`hardfloat/src/main/scala/`) 별도로 emit 하거나, 다른 .v 파일 안에서 module 선언을 grep 해서 찾을 수 있음.

## Troubleshooting

| 증상 | 원인 / 해결 |
|---|---|
| `java: command not found` | JAVA_HOME / PATH 설정 안 됨. 환경변수 확인. 새 터미널 열기. |
| `sbt: command not found` | sbt 설치 안 됨 or PATH 에 없음. |
| `[error] sbt.librarymanagement.ResolveException: Error downloading edu.berkeley.cs:chisel3_2.13:3.5.6` | 인터넷 차단 / 회사 방화벽. proxy 설정 필요 (`~/.sbt/repositories`). |
| `[error] Module reference is recursive` | Chisel 버전 mismatch. `build.sbt` 의 `scalaVersion`, Chisel 버전 확인. |
| Generated `.v` 의 module 이름이 mangled | Chisel param hash. 파일 안 `module XYZ_1` line 직접 보고 RTL 의 instantiation 명칭 맞추기. 또는 `--no-dedup` flag (Chisel 3.6+) 로 hash 비활성화. |
| 의존 module 누락 (`cannot find module ...`) | helper class 가 inline 안 됨. Scala 파일 추가로 emit 또는 grep 해서 누락 module 찾기. |

## 대안: 미리 만들어진 Verilog 받기

매번 sbt 빌드를 회피하고 싶으면:
1. 한 번 빌드해서 `third_party/berkeley-hardfloat/*.v` 를 만든 뒤 git commit.
2. 이후엔 sbt 환경 없이도 사용 가능 (`third_party/` 안의 .v 파일만 있으면 됨).
3. HardFloat 업데이트 필요 시에만 sbt 환경에서 재생성.

## 참고 자료

- [Chisel 3 docs](https://www.chisel-lang.org/chisel3/)
- [HardFloat README](https://github.com/ucb-bar/berkeley-hardfloat)
- [Recoded FP format 설명](https://github.com/ucb-bar/berkeley-hardfloat#recoded-format)
- [sbt manual](https://www.scala-sbt.org/1.x/docs/)
