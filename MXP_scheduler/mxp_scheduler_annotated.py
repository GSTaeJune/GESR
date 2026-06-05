# MXP_scheduler/mxp_scheduler_annotated.py
"""MXP_scheduler — 주석판(annotated) 쌍둥이 파일.

이 파일은 `mxp_scheduler.py`(표준판)의 **로직을 한 글자도 바꾸지 않고** 그대로 복제하되,
모든 수식/수량의 의미를 한국어로 줄단위 설명한 것이다. 학습/리뷰용으로 위에서 아래로
읽으면 비용 모델 전체를 이해할 수 있게 만든 독립 실행본이다.

설계 불변식:
  - 표준판과 **byte-identical** 한 evaluate()/optimize() 결과를 내야 한다.
  - 두 파일이 어떤 mapping 에서라도 어긋나면 `mxp_scheduler.py::crosscheck()` 가 실패한다.
  - 그래서 이 파일은 절대 `mxp_scheduler` 를 import 하지 않는다(독립 standalone).
    표준판의 수식을 잘못 베끼면 crosscheck 가 잡아준다 — 수식을 임의로 단순화/창작 금지.

Spec: docs/superpowers/specs/2026-06-04-mxp-scheduler-design.md
stdlib only.
"""
from dataclasses import dataclass, field
import itertools

# ── 모듈 상수 ────────────────────────────────────────────────────────────────
TILE = 32                # 물리 시스톨릭 어레이 한 변 = 공간(spatial) 매핑 (HW 고정값).
                         #   모든 차원(M,K,N)은 32 의 배수여야 하며, wbits 도 32×32 타일 단위 맵.
FP32_BITS = 32           # FP32 psum 워드 폭. C(출력)의 DRAM 트래픽과 on-chip 저장 둘 다 32-bit.
DEFAULT_COEFFS = {"dram": 200.0, "onchip": 6.0, "mac": 1.0, "rmw": 5.0}
                         #   에너지 가중치(coefficient): DRAM 전송 1bit=200, on-chip 버퍼 접근 1bit=6,
                         #   MAC 1회=1, RMW(dequant+FP32 add) 1회=5. DRAM 이 압도적으로 비쌈.


def divisors(n):
    # n 의 모든 약수(1..n)를 오름차순 리스트로. blocking factor 후보 생성에 쓰임.
    # 약수만 허용해야 타일 수 T 를 정확히 나눠떨어지게 분할(out*inn==T) 할 수 있다.
    return [d for d in range(1, n + 1) if n % d == 0]


# ── 하드웨어 기술서 ──────────────────────────────────────────────────────────
@dataclass
class HW:
    bank_size: int          # 뱅크당 워드 수 (on-chip SRAM 용량의 한 축).
    banks: int              # 뱅크 개수.
    dram_bw: float          # DRAM 버스 대역폭 = DRAM 사이클당 bit 수 (버스 throughput).
    word_bits: int = 32     # FP32 psum 워드 폭 (용량 계산 단위).
    freq_ratio: float = 1.0  # on-chip 사이클 / DRAM 사이클 (= f_chip / f_dram).
                             #   DRAM 전송시간을 on-chip 사이클(compute_work 의 단위)로 환산하는 비율.
    coeffs: dict = field(default_factory=lambda: dict(DEFAULT_COEFFS))
                             #   에너지 계수. dataclass mutable default 회피 위해 매 인스턴스 복제본 생성.

    def __post_init__(self):
        # 용량을 이루는 값들은 모두 양수여야 함. 0/음수면 cap_bits 가 0 이하가 되어
        # feasible 판정이 무의미해진다.
        for name, v in (("bank_size", self.bank_size), ("banks", self.banks),
                        ("word_bits", self.word_bits)):
            if v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")
        # dram_bw 는 stall_fill/_stall_of_order 에서 모든 fetch 의 분모. 0 이면 div-by-zero,
        # 음수면 사이클이 음수가 되어 불가능한(impossibly low) 비용이 나옴 → 양수 강제.
        if self.dram_bw <= 0:
            raise ValueError(f"dram_bw must be positive (bits/cycle), got {self.dram_bw}")
        # freq_ratio 는 DRAM 전송시간을 on-chip 사이클로 환산하는 인수. 0/음수면 환산이
        # 0 이 되거나 부호가 뒤집힌다 → 양수 강제.
        if self.freq_ratio <= 0:
            raise ValueError(f"freq_ratio must be positive (f_chip/f_dram), got {self.freq_ratio}")
        # coeffs 는 optimizer 의 목적함수에 들어간다: 음수 계수면 트래픽 많은 매핑이 음수 에너지로
        # 이기고, 키가 빠지면 energy_breakdown 깊숙한 곳에서 crash. 4개 키 모두 존재 + 각각
        # 비음수(non-negative) 숫자임을 강제.
        for k in ("dram", "onchip", "mac", "rmw"):
            if k not in self.coeffs:
                raise ValueError(f"coeffs missing required key {k!r}")
            if not isinstance(self.coeffs[k], (int, float)) or self.coeffs[k] < 0:
                raise ValueError(f"coeffs[{k!r}] must be a non-negative number, got {self.coeffs[k]!r}")

    @property
    def eff_bw(self):
        # on-chip 사이클 기준 유효 DRAM 대역폭 (on-chip 사이클당 bit):
        #   bits/DRAM사이클 × (DRAM사이클/on-chip사이클) = dram_bw / freq_ratio.
        # freq_ratio=1 (동일 클럭) 이면 eff_bw == dram_bw (하위호환).
        return self.dram_bw / self.freq_ratio

    @property
    def cap_bits(self):
        # on-chip 총 저장 용량(bit) = 워드수 × 뱅크수 × 워드폭. feasibility 의 상한.
        return self.bank_size * self.banks * self.word_bits


# ── 워크로드 기술서 ──────────────────────────────────────────────────────────
@dataclass
class Work:
    M: int
    K: int
    N: int
    wbits: list             # MT×KT, 32×32 타일당 평균 weight 비트수. 각 값 ∈ [2,8].
    act_bits: int           # 레이어 전체 균일 activation 정밀도 (2/4/8).

    def __post_init__(self):
        # M/K/N 은 TILE 의 양수 배수여야 한다: wbits 가 32×32 타일 단위 맵이라 마지막
        # 부분타일을 표현할 수 없고, 모든 카운트를 한 기준(raw M*K*N == 타일 MT*TILE*…)에
        # 맞춰 raw-vs-tiled 불일치를 제거한다.
        for name, v in (("M", self.M), ("K", self.K), ("N", self.N)):
            if v <= 0 or v % TILE != 0:
                raise ValueError(f"{name} must be a positive multiple of TILE={TILE}, got {v}")
        if self.act_bits not in (2, 4, 8):
            raise ValueError(f"act_bits must be 2/4/8, got {self.act_bits}")
        # wbits 모양 검증: MT 행 × KT 열.
        if len(self.wbits) != self.MT or any(len(r) != self.KT for r in self.wbits):
            raise ValueError(f"wbits must be {self.MT}x{self.KT}")
        # 타일당 '평균' weight 비트 — 분수일 수 있으나 [2,8] 범위 안이어야 함.
        if any(not (2 <= b <= 8) for row in self.wbits for b in row):
            raise ValueError("each wbits entry (average weight bits) must be in [2, 8]")

    @property
    def MT(self):
        return -(-self.M // TILE)   # ceil(M/TILE). 음수나눗셈 트릭으로 올림.

    @property
    def KT(self):
        return -(-self.K // TILE)

    @property
    def NT(self):
        return -(-self.N // TILE)

    @property
    def total_w_bits(self):
        # 전체 weight 비트수 = (타일당 32×32 element) × (모든 타일 평균비트 합).
        return TILE * TILE * sum(sum(row) for row in self.wbits)


# ── 매핑(루프 순서 + blocking) 기술서 ───────────────────────────────────────
@dataclass(frozen=True)
class Mapping:
    perm: tuple             # ("M","K","N") 의 순열. perm[0] = 최외곽(outermost) 루프.
    m_in: int               # 차원별 resident(내부, inner) 타일 수.
    k_in: int
    n_in: int

    def __post_init__(self):
        # _blocks/_stall_of_order 가 idx["M"/"K"/"N"] 로 인덱싱하므로, 순열이 아니면
        # 늦은 KeyError 나 조용한 오작동이 난다 → 생성 시점에 검증.
        if tuple(sorted(self.perm)) != ("K", "M", "N"):
            raise ValueError(f"perm must be a permutation of ('M','K','N'), got {self.perm}")


def _out_in(m, w):
    """차원별 (outer, inner) 분해 dict 반환.

    blocking factor 는 자기 타일 수를 정확히 나눠떨어져야 한다. gen_mappings 는 약수만
    내보내지만, 손으로 만든 잘못된 Mapping 은 floor-division 으로 outer 가 0 이 되어
    음수 Cr → total<0 으로 optimizer 를 오염시킬 수 있어 여기서 막는다."""
    for dim, T, f in (("M", w.MT, m.m_in), ("K", w.KT, m.k_in), ("N", w.NT, m.n_in)):
        if f < 1 or T % f != 0:
            raise ValueError(f"{dim} blocking factor {f} must be a divisor of tile count {T}")
    inn = {"M": m.m_in, "K": m.k_in, "N": m.n_in}                 # resident(내부) 타일 수
    out = {"M": w.MT // m.m_in, "K": w.KT // m.k_in, "N": w.NT // m.n_in}  # 외부 루프 반복 수
    return out, inn


def gen_mappings(w):
    # 전수 탐색용 후보 생성: 6 순열 × (각 차원 타일수의 약수)^3.
    ms = []
    for perm in itertools.permutations(("M", "K", "N")):
        for mi in divisors(w.MT):
            for ki in divisors(w.KT):
                for ni in divisors(w.NT):
                    ms.append(Mapping(perm=perm, m_in=mi, k_in=ki, n_in=ni))
    return ms


def footprint_bits(m, w):
    # 동시에 on-chip 에 상주해야 하는 작업집합 저장량(spec §6.1). a_blk·c_blk 은 (매핑 상수인)
    # resident A·C 창; foot_w 는 RESIDENT-WINDOW W 비트를 외부 블록들에 대해 max 한 값
    # (모든 타일의 GLOBAL 최대값이 아님 — 그러면 과대계상으로 저비트 resident mixed-precision
    # 매핑을 과도 prune 한다). _blocks 를 도는 김에 blocking factor 도 (_out_in 으로) 검증해
    # feasible() 이 잘못된 매핑을 dram_bits 와 일관되게 거부하게 한다.
    blocks = list(_blocks(m, w))
    _, _, a_blk, _w_blk0, c_blk = blocks[0]
    foot_w = max(b[3] for b in blocks)   # b[3] = w_blk = resident M_in×K_in 창 비트
    return a_blk + foot_w + c_blk


def feasible(m, w, hw):
    # resident 작업집합이 on-chip 용량 안에 들어가는가.
    return footprint_bits(m, w) <= hw.cap_bits


def dram_bits(m, w):
    """순서 의존(order-DEPENDENT) DRAM 트래픽(bit). 이 루프 순서에서 실제로 일어나는
    load/store 만 센다. stall 모델과 '같은' 외부블록 walk(_blocks) 위에서 세므로 A/W DRAM
    물량이 stall_fill 의 총 입력 fetch 와 구조적으로 일치한다(두 모델 불일치 없음).

    spec §6.2 의 순서 독립 reload factor (A=M_out, W=N_out, C=K_out)를 대체한다. 그것은
    재사용 차원이 최외곽인 worst-case 에서만 성립. 2026-06-05 결정: 에너지는 실제 루프 순서를
    반영해야 optimizer 가 순서를 고를 수 있다.

      A (=activation[K,N], M 에 대해 재사용): (K,N) 외부 인덱스가 바뀔 때 (재)load.
        (첫 블록은 항상 load) M 이 K,N 보다 안쪽이면 A 가 resident 로 남아 load 가 적다.
      W (=weight[M,K], N 에 대해 재사용): (M,K) 외부 인덱스가 바뀔 때 (재)load.
      C (=output[M,N], K 에 대해 reduce): K 가 '최내곽'이면 psum 이 완전 resident 로
        누적 → 마지막 1회 write, reload 없음(output-stationary). 아니면 C 타일이 evict 됐다가
        외부-K 마다 한 번씩 재접근 → spill (K_out 회 write, K_out-1 회 read; 첫 접촉은 zero-init)."""
    out, _ = _out_in(m, w)
    a = w_dram = 0.0
    prev = None
    for idx, _compute, a_blk, w_blk, _c_blk in _blocks(m, w):
        # A 는 (K,N) 으로 인덱싱 → (K,N) 외부 인덱스가 바뀌면 재load (첫 블록 prev=None 도 load).
        if prev is None or (idx["K"], idx["N"]) != (prev["K"], prev["N"]):
            a += a_blk                                  # A reload (A indexed K,N)
        # W 는 (M,K) 로 인덱싱 → (M,K) 외부 인덱스가 바뀌면 재load.
        if prev is None or (idx["M"], idx["K"]) != (prev["M"], prev["K"]):
            w_dram += w_blk                             # W reload (W indexed M,K)
        prev = idx
    full_c = w.M * w.N * FP32_BITS                       # C 전체를 1회 쓰는 비트수 (M*N*32)
    # C 타일은 모든 외부-K 방문에 걸쳐 누적된다. evict 됐다가 재접근되지 않는 한 누적 내내
    # resident(최종 1회 write, spill 없음). 재접근은 K 가 tiled(K_out>1) 이고 K 보다 '안쪽'
    # 루프 중 비자명(out>1)한 게 있어 다른 (M,N) 타일이 K 방문 사이에 끼어들 때만 발생.
    # K 최내곽(안쪽 루프 없음) 과 K_out==1 둘 다 output-stationary. (단순한 "K 최내곽" 만으로는
    # K 가 외부지만 안쪽 루프가 전부 자명한 경우를 과대계상한다.)
    k_pos = m.perm.index("K")
    interleaved = any(out[d] > 1 for d in m.perm[k_pos + 1:])   # K 보다 안쪽에 비자명 루프?
    if out["K"] > 1 and interleaved:
        cw = full_c * out["K"]                          # 외부-K 방문마다 partial write 1회
        cr = full_c * (out["K"] - 1)                    # 직전 partial reload; 첫 접촉은 zero-init
    else:
        cw, cr = full_c, 0                              # output-stationary: psum spill 없음
    return {"A": a, "W": w_dram, "Cw": cw, "Cr": cr, "total": a + w_dram + cw + cr}


_DISP = {8: 1, 4: 2, 2: 4}   # activation 모드별 col fire 당 RMW dispatch 수.
                             #   A8 은 col 당 1회, A4 는 2회, A2 는 4회 dispatch.


def mac_ops(w):
    # 이상적 MAC 연산 수 = M*K*N (정밀도와 무관한 곱셈누산 횟수).
    return w.M * w.K * w.N


def rmw_ops(w):
    # RMW(dequant INT→FP32 + FP32 add) 총 횟수 = 타일수 × (타일당 32×32 element) × dispatch.
    return w.MT * w.KT * w.NT * TILE * TILE * _DISP[w.act_bits]


def compute_work(w):
    # Σ_cube 32·b = TILE · NT · Σ_{mt,kt} wbits[mt][kt]  (순서 독립 이상적 하한).
    # int() 금지: 분수 평균 wbits 일 때 이 값은 _blocks 의 블록별 compute 합과 정확히 같아야
    # 한다. 안 그러면 actual_cycle(= compute_work + fill + stall)이 절단된 기준과 정확한
    # 기준을 섞게 된다.
    return TILE * w.NT * sum(sum(row) for row in w.wbits)


def onchip_bits(m, w):
    # 세 항 모두 on-chip 버퍼 접근이라 단일 "onchip" 에너지 계수를 공유(spec §7):
    # SA 가 읽는 a_rd, w_rd PLUS DRAM 에서 refill 한 데이터의 on-chip write.
    # 그 refill 의 DRAM-측 전송 에너지는 dram_bits("dram" 계수)에서 따로 센다 — 여기는 on-chip 측.
    a_rd = (w.MT * w.KT * w.NT) * TILE * TILE * w.act_bits   # A 는 cube 당 1회 읽힘
    w_rd = w.NT * w.total_w_bits                              # W 타일은 n_t 당 1회 읽힘
    d = dram_bits(m, w)
    refill = d["A"] + d["W"] + d["Cr"]                        # DRAM→on-chip load (매핑 가변)
    return a_rd + w_rd + refill                               # 정확값(평균 wbits 분수면 분수)


def energy_breakdown(m, w, hw):
    # 4개 비용원의 에너지 = 각 카운트 × 해당 계수. total 은 합.
    c = hw.coeffs
    e_dram = dram_bits(m, w)["total"] * c["dram"]
    e_onchip = onchip_bits(m, w) * c["onchip"]
    e_mac = mac_ops(w) * c["mac"]
    e_rmw = rmw_ops(w) * c["rmw"]
    return {"dram": e_dram, "onchip": e_onchip, "mac": e_mac, "rmw": e_rmw,
            "total": e_dram + e_onchip + e_mac + e_rmw}


def _blocks(m, w):
    """perm 순서(최외곽 먼저)로 외부 블록마다 (idx, compute_blk, a_blk, w_blk, c_blk) yield.
    idx        = 차원별 외부 인덱스 dict;
    compute_blk = 32*N_in*Σwbits(resident M_in×K_in 창);
    a_blk      = A fetch 비트 (매핑 상수: 매 블록 같은 inner A 타일);
    w_blk      = 이 블록의 W fetch 비트 (가변 — 블록의 resident wbits 에 의존);
    c_blk      = resident C psum 창 비트 (매핑 상수): 그 (M,N) 타일이 evict 되면 spill,
                 재진입하면 reload."""
    out, inn = _out_in(m, w)
    a_blk = inn["K"] * inn["N"] * TILE * TILE * w.act_bits   # 매핑 상수, 매 블록 동일
    c_blk = inn["M"] * inn["N"] * TILE * TILE * FP32_BITS    # 매핑 상수 resident C 창
    ranges = [range(out[d]) for d in m.perm]                 # perm 순서대로 외부 루프 범위
    for combo in itertools.product(*ranges):                 # 외부 인덱스 모든 조합 (사전식)
        idx = dict(zip(m.perm, combo))
        mo, ko = idx["M"], idx["K"]
        # 이 블록의 resident M_in×K_in 창에 속한 타일들의 평균 wbits 합.
        w_sum = sum(w.wbits[mt][kt]
                    for mt in range(mo * inn["M"], (mo + 1) * inn["M"])
                    for kt in range(ko * inn["K"], (ko + 1) * inn["K"]))
        # compute_blk = 32 · N_in · Σwbits ;  w_blk = Σwbits · 32 · 32 (타일당 element).
        yield idx, TILE * inn["N"] * w_sum, a_blk, w_sum * TILE * TILE, c_blk


def _stall_of_order(blocks, bw):
    """주어진 블록 ORDER 의 총 stall. 대역폭 bw 의 단일 DRAM 채널을 모든 트래픽이 공유:
    입력 fetch(A,W reload) + C psum reload-read + C psum spill-write.
    각 블록의 compute 창이 그 주변 전송을 hide 한다 — 다음 블록의 입력+C reload(연산 전에
    도착해야 함) PLUS 이전 블록의 C spill-write(연산 후 drain). hide 못 한 전송시간이 stall.
    마지막 블록의 C write 는 뒤따르는 compute 가 없어 hide 불가 → trailing drain(항상 가산).

    A 는 (K,N) 변경 시 reload, W 는 (M,K) 변경 시. C 는 spill 후 (M,N) 타일 재진입 시 reload,
    (M,N) 타일을 떠날 때(매 episode 끝, 최종 write 포함) spill. (입력만 보던 spec §8 stall 을
    대체 — 이제 C psum 대역폭도 모델링.)"""
    n = len(blocks)
    if n == 0:
        return 0.0
    seen = set()              # 지금까지 접촉한 (M,N) 타일 집합 (재진입 = spill reload 판정)
    prev = None               # 직전 블록 idx
    prev_compute = 0.0        # 직전 블록 compute 사이클 (이번 전송을 hide 함)
    total = 0.0
    for idx, compute_blk, a_blk, w_blk, c_blk in blocks:
        mn = (idx["M"], idx["N"])
        if prev is not None:
            prev_mn = (prev["M"], prev["N"])
            fetch = 0.0
            if (idx["K"], idx["N"]) != (prev["K"], prev["N"]):
                fetch += a_blk                              # A reload (A indexed K,N)
            if (idx["M"], idx["K"]) != (prev["M"], prev["K"]):
                fetch += w_blk                              # W reload (W indexed M,K)
            if mn != prev_mn:
                fetch += c_blk                              # 이전 블록이 C 타일을 떠남 → spill drain
                if mn in seen:
                    fetch += c_blk                          # spill 된 C 타일 재진입 → reload read
            # 전송시간(fetch/bw)에서 직전 compute 로 hide 한 만큼 빼고 남는 게 stall (음수면 0).
            total += max(0.0, fetch / bw - prev_compute)
        seen.add(mn)
        prev = idx
        prev_compute = compute_blk
    total += blocks[-1][4] / bw                             # 마지막 타일의 trailing C drain (c_blk/bw)
    return total


def stall_fill(m, w, hw):
    """외부 반복 블록에 대한 sequence-aware stall. DRAM 대역폭은 A/W 입력 fetch 와 C psum
    spill/reload 가 공유(_stall_of_order 참조). fill = 첫 블록의 입력 fetch 시간(hide 불가
    startup); stall = 나머지(블록별 hide 안 된 전송 + trailing C drain)."""
    blocks = list(_blocks(m, w))
    bw = hw.eff_bw                                          # on-chip 사이클 기준 DRAM 대역폭
    _, _, a0, w0, _c0 = blocks[0]
    fill = (a0 + w0) / bw                                   # 첫 블록 C read 는 0 (첫 접촉)
    return _stall_of_order(blocks, bw), fill


def actual_cycle(m, w, hw):
    # 실제 사이클 = 이상적 compute_work + 시작 fill + 가려지지 않은 stall.
    stall, fill = stall_fill(m, w, hw)
    return float(compute_work(w)) + fill + stall


def evaluate(m, w, hw):
    # 한 매핑의 전 지표를 dict 로. optimize/report 가 이 결과를 소비.
    feas = feasible(m, w, hw)
    eb = energy_breakdown(m, w, hw)
    stall, fill = stall_fill(m, w, hw)
    cw = compute_work(w)
    return {
        "mapping": m,
        "feasible": feas,
        "energy": eb["total"],
        "energy_breakdown": eb,
        "dram": dram_bits(m, w),
        "compute_work": cw,
        "stall": stall,
        "fill": fill,
        "actual_cycle": float(cw) + fill + stall,
    }


def optimize(w, hw, max_cycle=None):
    """(perm × blocking) 전수 탐색. feasible 결과를 energy 오름차순, 동률은 actual_cycle
    오름차순으로 정렬. max_cycle 주어지면 정렬 '전에' actual_cycle <= max_cycle 로 추가 필터.

    energy 는 순서 의존(DRAM 트래픽이 루프 순서에 의존)이라 순열마다 대개 다르다. 그래도 두
    매핑이 energy 동률이면 actual_cycle 오름차순이 (gen_mappings 삽입순서가 아니라) 결정적으로
    동률을 깬다. 동일 energy 면 적은 사이클이 엄밀히 더 좋으므로 올바른 키다."""
    results = [evaluate(m, w, hw) for m in gen_mappings(w)]
    results = [r for r in results if r["feasible"]]
    if max_cycle is not None:
        results = [r for r in results if r["actual_cycle"] <= max_cycle + 1e-9]
    results.sort(key=lambda r: (r["energy"], r["actual_cycle"]))
    return results


def lpt_headroom(m, w, hw):
    """Headroom 지표(spec §9.2): 자연 perm-순서 stall vs 외부 블록을 Longest-Processing-Time
    먼저(블록 compute 내림차순) 재정렬한 stall. 둘 다 full 대역폭 모델(A/W fetch + C psum
    spill/reload) 사용. 여기선 LPT 가 drop-in 최적이 아니다(타일이 재사용으로 결합돼 있고
    재정렬이 C residency 를 교란) → headroom 이 <= 0 일 수도 있음."""
    blocks = list(_blocks(m, w))
    natural = _stall_of_order(blocks, hw.eff_bw)
    lpt = _stall_of_order(sorted(blocks, key=lambda b: -b[1]), hw.eff_bw)  # b[1] = compute_blk 내림차순
    return {"natural_stall": natural, "lpt_stall": lpt, "headroom": natural - lpt}


def report(ranked, w, hw, top=10):
    # 상위 top 매핑을 표 형태 텍스트로. CLI 출력용.
    lines = [f"# GEMM ({w.M}x{w.K}x{w.N})  cap_bits={hw.cap_bits}  dram_bw={hw.dram_bw}",
             f"{'perm':<10}{'m_in':>5}{'k_in':>5}{'n_in':>5}{'energy':>16}{'actual_cycle':>16}{'stall':>12}"]
    for r in ranked[:top]:
        m = r["mapping"]
        lines.append(f"{''.join(m.perm):<10}{m.m_in:>5}{m.k_in:>5}{m.n_in:>5}"
                     f"{r['energy']:>16.0f}{r['actual_cycle']:>16.1f}{r['stall']:>12.1f}")
    return "\n".join(lines)


def pareto_front(ranked):
    """(energy, actual_cycle) 상의 비지배(non-dominated) 집합. energy 오름차순 정렬 후 반환.
    (energy, cycle) 로 정렬하면, 한 점이 front 에 드는 조건은 '지금까지 유지된 더 낮은 energy
    의 모든 점보다 cycle 이 엄밀히 개선'되는 것뿐이다."""
    pts = sorted(ranked, key=lambda r: (r["energy"], r["actual_cycle"]))
    front = []
    best_cycle = float("inf")
    for r in pts:
        if r["actual_cycle"] < best_cycle - 1e-9:
            front.append(r)
            best_cycle = r["actual_cycle"]
    return front


def tradeoff(w, hw):
    """OFF = 전역 최소 energy 매핑(cycle 무시). ON = 최소 actual_cycle 매핑들 중 최소 energy
    (가장 빠른 스케줄, 그 속도에서 가장 싼 energy)."""
    ranked = optimize(w, hw)
    if not ranked:
        raise ValueError("no feasible mapping for this HW (check capacity)")
    off = ranked[0]
    min_cycle = min(r["actual_cycle"] for r in ranked)
    on = min((r for r in ranked if r["actual_cycle"] <= min_cycle + 1e-9),
             key=lambda r: r["energy"])
    return {"off": off, "on": on, "pareto": pareto_front(ranked)}


# ── 주석판 전용: 단계별 중간값을 인쇄하는 explain ────────────────────────────
def explain(m, w, hw):
    """한 매핑의 비용을 단계별 중간값으로 풀어서 인쇄(학습/디버그용). 표준판의 수식을
    그대로 호출해 같은 숫자를 보여준다. 출력에는 반드시 "compute_work" 와 "stall" 문자열
    포함(테스트 게이트)."""
    out, inn = _out_in(m, w)
    print("=== MXP_scheduler explain (annotated) ===")
    print(f"workload : M={w.M} K={w.K} N={w.N}  (MT={w.MT} KT={w.KT} NT={w.NT})  act_bits={w.act_bits}")
    print(f"hardware : cap_bits={hw.cap_bits}  dram_bw={hw.dram_bw}  freq_ratio={hw.freq_ratio}  eff_bw={hw.eff_bw}")
    print(f"mapping  : perm={''.join(m.perm)}  inner(M,K,N)=({inn['M']},{inn['K']},{inn['N']})  "
          f"outer(M,K,N)=({out['M']},{out['K']},{out['N']})")
    print(f"feasible : {feasible(m, w, hw)}   (footprint_bits={footprint_bits(m, w)} <= cap_bits={hw.cap_bits})")

    # 블록 walk: 외부 블록 수와 블록별 compute 합이 compute_work 와 일치함을 보여준다.
    blocks = list(_blocks(m, w))
    print(f"blocks   : {len(blocks)} outer block(s) in perm order")
    block_compute_sum = sum(b[1] for b in blocks)
    cw = compute_work(w)
    print(f"compute_work : {cw}  (= TILE*NT*Σwbits ; Σ per-block compute = {block_compute_sum})")

    # DRAM 트래픽 분해 (순서 의존).
    d = dram_bits(m, w)
    print(f"dram_bits    : A={d['A']}  W={d['W']}  Cw={d['Cw']}  Cr={d['Cr']}  total={d['total']}")

    # on-chip / mac / rmw 카운트.
    print(f"onchip_bits  : {onchip_bits(m, w)}")
    print(f"mac_ops      : {mac_ops(w)}   rmw_ops : {rmw_ops(w)}")

    # 에너지 분해.
    eb = energy_breakdown(m, w, hw)
    print(f"energy       : dram={eb['dram']}  onchip={eb['onchip']}  mac={eb['mac']}  rmw={eb['rmw']}  "
          f"total={eb['total']}")

    # 사이클: fill + stall + compute = actual_cycle.
    stall, fill = stall_fill(m, w, hw)
    print(f"stall / fill : stall={stall}  fill={fill}   (DRAM bw shared by A/W fetch + C spill/reload)")
    print(f"actual_cycle : {actual_cycle(m, w, hw)}  (= compute_work {cw} + fill {fill} + stall {stall})")

    # LPT headroom 지표.
    h = lpt_headroom(m, w, hw)
    print(f"lpt_headroom : natural_stall={h['natural_stall']}  lpt_stall={h['lpt_stall']}  "
          f"headroom={h['headroom']}")


def _load_wbits(path, MT, KT):
    import json
    with open(path) as f:
        wb = json.load(f)
    if len(wb) != MT or any(len(r) != KT for r in wb):
        raise ValueError(f"wbits file must be {MT}x{KT}")
    return wb


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="MXP_scheduler (annotated) — 단계별 explain + 표준 CLI 와 동일 결과")
    p.add_argument("--explain", action="store_true",
                   help="optimize 의 최적 매핑(없으면 첫 매핑)에 대해 단계별 중간값 인쇄")
    p.add_argument("--M", type=int); p.add_argument("--K", type=int); p.add_argument("--N", type=int)
    p.add_argument("--bank-size", type=int, default=1024)
    p.add_argument("--banks", type=int, default=32)
    p.add_argument("--dram-bw", type=float, default=64.0, help="bits per DRAM cycle")
    p.add_argument("--freq-ratio", type=float, default=1.0,
                   help="on-chip cycles per DRAM cycle (f_chip/f_dram); 1.0 = same clock")
    p.add_argument("--act", type=int, default=8)
    p.add_argument("--bits-file", help="JSON MT x KT avg-weight-bits; default all=act")
    p.add_argument("--max-cycle", type=float, default=None)
    p.add_argument("--coeffs", help="JSON override of energy coeffs")
    a = p.parse_args(argv)
    if not (a.M and a.K and a.N):
        p.error("provide --M --K --N")
    MT, KT = -(-a.M // TILE), -(-a.K // TILE)
    wbits = _load_wbits(a.bits_file, MT, KT) if a.bits_file else [[a.act] * KT for _ in range(MT)]
    coeffs = dict(DEFAULT_COEFFS)
    if a.coeffs:
        import json
        coeffs.update(json.load(open(a.coeffs)))
    w = Work(M=a.M, K=a.K, N=a.N, wbits=wbits, act_bits=a.act)
    hw = HW(bank_size=a.bank_size, banks=a.banks, dram_bw=a.dram_bw,
            freq_ratio=a.freq_ratio, coeffs=coeffs)
    ranked = optimize(w, hw, max_cycle=a.max_cycle)
    if a.explain:
        # 최적 매핑(없으면 첫 후보)을 골라 단계별로 설명.
        m = ranked[0]["mapping"] if ranked else gen_mappings(w)[0]
        explain(m, w, hw)
    else:
        print(report(ranked, w, hw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
