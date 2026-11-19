import asyncio

from app.chat import ChatService
from app.config import Settings
from app.verify import verify_answer

CONTEXT = """[1] Neptune: SAR 880 million (GBP 180 million), handover December 2027.
WASALT LISTINGS (median SAR 1,600,000 (~1.60M), range SAR 500,000 - SAR 2,000,000 (~2.00M))
[L1] Villa for sale | Al Marjan, Riyadh | SAR 1,600,000 (~1.60M) | 10 bed | 540 sqm
[L2] Land for sale | King Fahad Dist, Dammam | SAR 1,187,040 (~1.19M)"""


def test_rounded_figures_from_sources_pass():
    answer = ("Median ~SAR 1.6M, range SAR 500K-2M [L1]. Land at SAR 1.19M [L2]. "
              "Neptune is SAR 880 million, handover 2027 [1]. About SAR 2,963/sqm.")
    v = verify_answer(answer, CONTEXT)
    assert v.unsupported == [] and len(v.checked) == 6     # per-m2 values are derived, not checked


def test_invented_figures_are_flagged():
    v = verify_answer("Neptune villas cost SAR 3.2M and complete in 2028 [1].", CONTEXT)
    assert v.unsupported == ["3.2M", "2028"]


def test_arabic_digits():
    v = verify_answer("الفيلا بسعر ١٬٦٠٠٬٠٠٠ ريال [L1]، والوسيط ٢٫٣ مليون ريال.", CONTEXT)
    assert v.unsupported == ["2.3 مليون"]


def test_uncited_answers_are_not_checked():
    assert verify_answer("15% of 2.4 million is 360,000.", CONTEXT).checked == []


def test_chat_emits_verification(kb):
    from tests.test_llm_chat import MODELS, FakeResp, collect, make_client, sse
    llm = make_client([FakeResp(200, sse("Trump Tower Jeddah completes in 2031 [1]."))], MODELS)
    asyncio.run(llm.refresh_models())
    evs = asyncio.run(collect(ChatService(Settings(), kb, llm).answer("When is Trump Tower Jeddah done?", [])))
    check = next(e for e in evs if e["type"] == "verify")
    assert check["unsupported"] == ["2031"]


def test_thresholds_are_not_claims():
    v = verify_answer("For luxury options (above 5 million) see Wasalt [L1]. Budgets up to SAR 3M work too.", CONTEXT)
    assert v.unsupported == []
    assert verify_answer("للخيارات الفاخرة (أعلى من 5 ملايين) [L1]", CONTEXT).unsupported == []
