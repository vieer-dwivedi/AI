from app.tools.erp_server import (calculate_penalty, check_inventory, compute_penalty,
                                  get_customer, get_order)


def test_zero_delay():
    assert compute_penalty(950000, 0, 2, 5.0, 25.0) == 0


def test_delay_inside_grace():
    assert compute_penalty(950000, 2, 2, 5.0, 25.0) == 0


def test_normal_penalty():
    assert compute_penalty(950000, 6, 2, 5.0, 25.0) == 190000.0


def test_cap_reached():
    assert compute_penalty(100000, 30, 2, 5.0, 10.0) == 10000.0


def test_get_order_scenario_1():
    o = get_order(48213)
    assert o["customer_id"] == "CUST-007" and o["value_inr"] == 950000.0
    assert o["shipment"]["tracking_id"] == "TRK-48213"


def test_get_order_unknown():
    assert get_order(1)["error"] == "ORDER_NOT_FOUND"


def test_get_customer_tier1():
    assert get_customer("CUST-007")["tier"] == 1


def test_inventory_pune():
    stock = check_inventory("SKU-001", "Pune")["stock"][0]
    assert stock["available"] == 400


def test_penalty_tool_scenario_1():
    assert calculate_penalty("CON-007", 6, 950000)["penalty_inr"] == 190000.0