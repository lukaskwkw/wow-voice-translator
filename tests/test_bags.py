"""Vendor sell prices: the 8th bag field, the junk total and the short coin form."""

from tests.test_assistant import make_assistant
from wvt import protocol as P
from wvt.assistant import _coins
from wvt.bags import parse_line


def bag_assistant(*lines):
    a, *_ = make_assistant()
    text = "\n".join(lines)
    pkt = P.Packet(type=P.TYPE_CHAT, seq=1, chat_kind=P.KIND_BAG, race="1:1/1", text=text)
    assert a.bags.on_packet(pkt) and a.bags.known
    return a


def test_a_7_field_line_has_no_price():
    item = parse_line("2589\t12\t1\t7\t5\t0\tLinen Cloth")
    assert item is not None and item.sell_price == 0 and item.name == "Linen Cloth"


def test_an_8_field_line_carries_the_vendor_price():
    item = parse_line("2589\t12\t1\t7\t5\t0\tLinen Cloth\t40")
    assert item is not None and item.sell_price == 40 and item.name == "Linen Cloth"


def test_garbage_in_the_8th_field_is_zero():
    item = parse_line("2589\t12\t1\t7\t5\t0\tLinen Cloth\tnope")
    assert item is not None and item.sell_price == 0


def test_coins_omit_zero_parts():
    assert _coins(2) == "2 c"
    assert _coins(1240) == "12 s 40 c"
    assert _coins(10300) == "1 zł 3 s"
    assert _coins(0) == "0 c"


def test_bag_junk_totals_the_prices_of_the_junk():
    a = bag_assistant("2589\t12\t1\t7\t5\t0\tLinen Cloth\t40",
                      "111\t3\t0\t7\t5\t0\tSzmaty\t33",
                      "112\t1\t0\t0\t0\t0\tButelka\t150")
    text = a._bag_junk()
    assert "Razem sprzedasz za około 2 s 49 c." in text


def test_bag_junk_without_prices_says_nothing_about_money():
    a = bag_assistant("111\t3\t0\t7\t5\t0\tSzmaty")
    text = a._bag_junk()
    assert "Razem sprzedasz za" not in text


def test_item_facts_show_the_vendor_price_of_a_bag_item():
    a = bag_assistant("111\t3\t0\t7\t5\t0\tSzmaty\t33")
    own = next(o for o in a._own_items() if o.name == "Szmaty")
    facts = a._item_facts("Szmaty", own)
    assert "Sprzedasz za: 33 c." in facts
