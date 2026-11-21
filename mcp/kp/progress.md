# KP MCP server: progress

Free, local server for **Krishnamurti Paddhati (KP) astrology**, mirroring `mcp/dasha` and `mcp/transit`
(FastMCP, `state` = birth details, lazy backend, tool wrapper turns engine errors into tool errors).
Decision (user, 2026-11-20): **no paid/hosted APIs** (RoxyAPI, Divine API rejected). Everything is computed locally with
Swiss Ephemeris (`pyswisseph`, already in `.venv`) plus our own KP tables. PyJHora is NOT needed for the core.

## Where I am
**Phases 1-5 DONE. Phase 6: backlog items 1-16, 19-21, 23-27 DONE (2026-11-20 / 21). Remaining: item 18 (golden charts: no complete published example chart with listed cusps / subs / dasha found yet), item 22 (orchestrator agent plugin `agents/kp.py`: needs the user's go-ahead), item 12b (Rule of Origin / Bosmia method, unverified), parked 17 and scope notes 28.**
Server: 43 tools, stdio and `--http` on 8002. `.venv/bin/python mcp/kp/test/test_kp.py` -> **447/447 checks pass** (~40 s); `test/try_client.py` drives the real server over stdio.
Files: `kp_tables, state, kp_core, kp_chart, kp_methods, kp_fruitful, kp_retro, kp_cusp, kp_csl_ext, kp_cil, kp_double, kp_transit, kp_longevity, kp_horary, kp_rectify, kp_twins, kp_muhurta, kp_report, kp_workflow, kp_place, kp_annual, kp_medical, kp_dasha, kp_moment, kp_server, main`, `test/{test_kp,smoke_chart,smoke_moment,try_client}.py`. Readme has a KP tool / option section.

### Known gaps / honest limits (to decide with the user)
- Fruitful-significator selection is DONE (2026-11-20, backlog item 3, `kp_fruitful.py`): four selectable methods; sub-lord permission can now filter periods. Default of `find_event_periods` is still no filter (`fruitful_method=None`).
- 4-step theory (Gondhalekar) and the 6-fold view are DONE (2026-11-20, `kp_methods.py`; judgment tools still use A-D); type-E significators are DONE (2026-11-20, backlog item 2; separate block, off by default).
- Retrograde rules are DONE (2026-11-20, backlog item 4, `kp_retro.py`): option `retrograde_rule` (default `report_only`), flags in periods and the CSL check, `find_stations` tool. No rule ever hard-denies; schools disagree.
- Cusp proximity (bhava chalit) is DONE (2026-11-20, backlog item 5, `kp_cusp.py`): option `cusp_orb_deg` (default off) reports `near_cusp`; `cusp_proximity_apply` (default off) feeds it into A-D. The 4-step school keeps its own fixed 3d20' cusp rule.
- CSL extensions are DONE (2026-11-20, backlog item 6, `kp_csl_ext.py`): option `csl_extension` (default `off`; `own_bhava` | `sub_chain` | `both`), separate `csl_extension` block and `extended_reading`; never changes A-D or the standard verdict. Not pure KP; the original KSK note could not be re-verified.
- Cuspal interlinks are DONE (2026-11-20, backlog item 7, `kp_cil.py`): tool `get_cuspal_interlinks` (facts + SSL reading beside the CSL verdict). Only the sub-sub-lord and CSL/star/sub link readings are encoded; Bhaskaran's four link kinds are not.
- Uranus/Neptune/Pluto: positions work (`include_outer`), but `PLANETS` loops ignore them in significators, as intended.
- Cuspal interlinks and the double transit are DONE (items 7, 8); see their notes. Double transit is a K.N. Rao / Parashari technique, not Krishnamurti's own rule (tool says so).
- Transit rule set and `verify_event_day` are DONE (item 9, `kp_transit.py`); `find_transit_triggers` (window scan) is unchanged.
- Mean node default; true-node sub lord flips are not flagged automatically.
- Rectification is a scan, not a proof; needs certain event dates.

## TO BE DONE (backlog, written 2026-11-20 after research round 3; nothing below is implemented yet)
Design rule for all of it: tools state facts and the rule applied, never predictions. KP schools disagree, so every school-dependent rule
becomes an explicit option (default = Krishnamurti Readers / "pure KP"), and the output says which rule was used. Details and sources: notes R15-R20 below.

### A. The items requested (priority 1; user re-confirmed 2026-11-20: 4-step + 6-fold + type E, partial fruitful-significator filtering, retrograde flags-only)
1. **[DONE 2026-11-20, see "Item 1 implementation notes" below] 4-step theory + 6-fold sub-level significator view** (R15, R10). New option `significator_method = "kp4level"` (today, default) | `"four_step"` | `"six_fold"`.
   - `four_step` (Gondhalekar / Shri Bhatt variant): step 1 planet (occupied + owned houses), step 2 its star lord, step 3 its sub lord, step 4 the star lord of that sub lord. Each house is tagged **primary (strong)** or **secondary (weak)**; only primary ones count for fruitful-significator selection (rule: "secondary significators are like a mirage"). Differs from KSK, where the sub lord is the end-result indicator; here the sub's star lord is.
   - Primary/secondary rules to code: planet in own star = primary for the house it occupies (even if others sit there); planet with no planet in its star = primary for occupied and owned houses only when those houses are empty-or-sole; planet on a cusp (within 3d20') = strong significator of that cusp; planets conjunct (3d20' orb) a strong planet become strong, else weak; Rahu/Ketu as star lord of a planet/sub give results of conjunction / aspect / star lord / sign lord in ascending order; ignore Uranus/Neptune/Pluto.
   - Rule 6 of that school: a planet that totally negates the event houses at step 3-4 will not fructify; if it is the cuspal sub lord the event never happens. Rule: transit uses only two steps (sign/star lord).
   - `six_fold`: KP Reader / EZine 128 style table adding planets in the **sub** of the occupant and of the owner to A-D (verify the exact column list against the EZine 128 table before coding). Tool `get_significators` gets `method`; `get_planet_significations` gets `include_sub_level`.
   - Tests: worked charts from the sources (the Four-Step page example: planet / star / sub / sub's star lines) as golden data; consistency (4-level A-D result must be a subset of four_step primary + secondary).
2. **[DONE 2026-11-20, see "Item 2 implementation notes"] Type-E significators** (conjunct / aspecting the significators; Krishnamurti's later addition, "E grade"). Option `include_type_e` (default off), listed as a separate `E` level, orb option (default 3d20' as in the 4-step school, or sign-based as today for node agents). Never mixed silently into A-D.
3. **[DONE 2026-11-20, see "Item 3 implementation notes"] Fruitful-significator filtering** (R8). Three selectable methods for `find_event_periods` / new `get_fruitful_significators(event, method)`:
   - `detrimental`: drop planets that signify the denial houses (12th-from group);
   - `sub_lord`: keep a planet only if its sub lord signifies the event houses and does not only signify denial houses (KP "permission");
   - `ruling_planets`: keep planets that are ruling planets at the moment of judgment (needs a moment; EZine 128 argues the sub cannot select fruitful significators without RPs, so default to RP-assisted).
   - Output = candidate lords with the reason each was kept/dropped, then joint periods filtered by them (today only "signifies a favourable house" is applied).
4. **[DONE 2026-11-20, see "Item 4 implementation notes"] Retrograde rules** (R16). Today only a flag. Add `retrograde_rule` option, default `report_only`:
   - `ksk_horary_cusp`: apply only to the cuspal sub lord, and only if it is retrograde itself or in the star of a retrograde planet (the rule that several teachers restrict to horary);
   - `strict`: a planet in the star of a retrograde planet, or a retrograde dasha/bhukti lord, is flagged as "delayed / needs the planet to turn direct" in period tables;
   - report the next station date (retrograde to direct) with `find_stations(planet, window)` so the caller can see when a delayed result can mature.
   Schools disagree (three views recorded in KP literature, Kesav Ranjan Kar research): never default to hard denial.

### Item 1 implementation notes (done 2026-11-20)
- New file `kp_methods.py`; option `KPOptions.significator_method` (`kp4level` default | `four_step` | `six_fold`, aliases `4step`, `6fold`); `get_significators(state, method=None)` adds a `four_step` or `six_fold` block (the A-D tables are always returned); `get_planet_significations(state, planet, include_sub_level=False)` adds `sub_levels` + `signifies_6fold`. Judgment tools (`judge_event`, `find_event_periods` ...) are unchanged and still use A-D.
- 4-step rules coded, from the Gondhalekar pages (astrogle "Four Step Theory and Explanation", Scribd copies): rule 1 for steps 1 and 3 (own star -> occupied house primary; no other planet in its star -> primary; planets in its star -> secondary; owned houses primary only if no other planet in them), rule 2 for steps 2 and 4 (star lord's occupied house always primary; owned houses primary only when vacant), Rahu/Ketu agents (node primary -> agents judged by rule 1; node secondary -> agents secondary), planet within 3d20' of a cusp = primary for that cusp's house. Output per planet: the four steps with every house tagged primary/secondary plus reason, `primary_houses`, `secondary_houses`, `primary_by_step`; house view lists primary and secondary significators.
- 6-fold: columns planets in sub of occupant, A, B, C, planets in sub of owner, D (EZine 128 / Rangasamy). The source table's column letters are not fully legible, so columns are named, not lettered.
- Still open inside this item: upgrade of planets conjunct a strong planet (3d20') is only reported (`planets_conjunct_within_orb`), not applied; type-E upgrades belong to item 2; the "denial at steps 3-4" test belongs to item 3; no golden test from the published worked example (its chart data is not in the source), tests are structural (disjoint sets, 4-level result is a subset of primary+secondary, house/planet views agree, own-star and rule 2 invariants): 19 new checks.
- Interpretation risk: sources are summaries and differ in wording; the "own house shared with another planet" case follows the pages above. Treat this as the Gondhalekar school only.

### Item 2 implementation notes (done 2026-11-20)
- `kp_methods.type_e(chart, basis, orb)`; options `include_type_e` (default False), `type_e_basis` (`orb` default | `sign`), `type_e_orb_deg` (default 3d20'). `get_significators(state, method=None, include_type_e=None)` adds a separate `type_e` block (never merged into A-D; A-D output verified identical with it on or off).
- Rule coded: planet P is an E significator of every house signified (A-D) by a planet it conjoins or aspects, and of every cusp it conjoins or aspects. Houses already in P's own A-D are listed apart (`already_in_A_to_D`), `e_houses` holds only the new ones. Each entry carries its links (planet or cusp, relation) so it can be audited.
- Aspects: sign-based house aspects (7th for all; Mars 4/8, Jupiter 5/9, Saturn 3/10); in `orb` basis the aspect point must also lie within the orb of the exact angle. Rahu/Ketu only conjoin (same convention as node agents). Uranus/Neptune/Pluto ignored.
- Not done: the 4-step school's upgrade of weak E significators to primary (conjunction/aspect with cusp or planet); E is not used by judgment tools. Direction of the aspect (aspecting vs aspected) follows the backlog wording: P aspects Q. 13 new checks (structure, symmetry of orb conjunction, sign-basis = same sign, nodes never aspect, tools).
- Source wording is thin (EZine/summary pages call E "conjunct/aspecting the significators"); treat as one reading of it.

### Item 3 implementation notes (done 2026-11-20)
- New `kp_fruitful.py`: `parse_methods`, `analyze(chart, fav, den, methods, rp)`. Methods (comma-separate to combine; a planet must pass all): `detrimental`, `sub_lord`, `ruling_planets`, `four_step_primary` (added beyond the backlog because item 1's school says only primary significators count; it also drops a planet whose steps 3-4 primary houses touch only denial houses, the school's "rule 6"). Aliases: `rp`, `sub`, `4step`, `primary`.
- Candidates = planets signifying a favourable house (A-D, `levels` option honoured). Output: every planet with `verdicts` per method and `reasons`, `fruitful` (ordered: favourable house via A/B first, fewer denial houses, more favourable houses), `dropped`, `not_candidates`, `ruling_planets_used`.
- New tool `get_fruitful_significators(state, event|custom houses, methods="detrimental", moment=None, levels=None)` (30 tools now). `ruling_planets` needs `moment` (place, optional date/time); no moment -> clean tool error. RPs come from `kp_moment.ruling_planets` with the birth's options.
- `find_event_periods` got `fruitful_method` (default None = old behaviour) and `moment`; every lord of a returned period must be fruitful (`find_joint_periods(..., allowed=)`); result has `fruitful_selection` and `fruitful_filter`. If `moment` is given and `ruling_planets` is not, the moment's RPs fill the `ruling_planet_lords` flag.
- Not done: `sub_lord` method uses the planet's own sub lord only (not sub lord's star lord); the RP method is strict (planet itself must be an RP; star/sub-of-RP are only flagged); schools disagree, nothing is the default. 17 new checks (197/197).

### Item 4 implementation notes (done 2026-11-20)
- New `kp_retro.py`; option `KPOptions.retrograde_rule` = `report_only` (default, facts only) | `ksk_horary_cusp` | `strict` (aliases `ksk`, `horary`, `report`). Rahu/Ketu (always retrograde as mean nodes) and Sun/Moon never trigger a rule. Status used is the natal one (speed sign in the chart).
- `ksk_horary_cusp`: only the cuspal sub lord is tested; triggered when the CSL is retrograde or in the star of a retrograde planet. `strict`: the same CSL test plus period lords (dasha / bhukti / antara ...) that are retrograde or in a retrograde star are flagged and the period is marked `delayed`.
- Where it shows: `csl_report` / `judge_event` carry `retrograde_check` (+ `retrograde_note` when triggered; the verdict itself is never changed, nothing is denied); `find_joint_periods` / `find_event_periods` rows carry `retrograde_flags` (facts, any rule) and `delayed` (strict only); with `strict`, `find_event_periods` adds `retrograde_transit_info` (is the planet retrograde in transit at the window start, next direct station); `analyze_joint_period` has a `retrograde` block.
- New tool `find_stations(moment, planet, start, end)` (31 tools): Mercury, Venus, Mars, Jupiter, Saturn; speed sign change bisected to about a second; returns `retrograde_at_start`, stations with sign / star / sub lord, `next_direct_station`, `next_retrograde_station` (searches up to about 800 days past the window). Checked against published 2026 dates: Mercury 26 Feb-21 Mar, 29 Jun-24 Jul, 24 Oct-13 Nov; Saturn 27 Jul-11 Dec.
- Not done / open: the horary-only reading is not enforced (the rule is applied to natal charts too when chosen); `strict` does not look at the sub lord of a period lord; no rule for the Moon's or Sun's transit; combust is still not treated. 22 new checks (219/219).

### Item 5 implementation notes (done 2026-11-20)
- New `kp_cusp.py` (pure helpers); options `KPOptions.cusp_orb_deg` (default `None` = off; 0 also off; max 15; usual 3.3333 = 3d20') and `cusp_proximity_apply` (default False; needs an orb, otherwise a validation error).
- Facts (any time an orb is set): each planet of the 9 within the orb BEHIND the next cusp gets a `behind_cusp` entry (house it sits in, the cusp it nears, distance); each planet within the orb PAST the cusp of its own house gets a `past_cusp` entry (strong significator of that house; distance 0 = exactly on the cusp). Wraps from house 12 to cusp 1. Uranus/Neptune/Pluto are not considered.
- Where it shows: `get_kp_chart` and `get_significators` get a `near_cusp` block (orb, applied flag, entries, rule, note); cusp rows get `planets_behind_cusp` / `planets_past_cusp`; planet significations get `near_cusp` and `signifies_with_near_cusp` (A-D houses plus the next house); house significators get `near_cusp_planets`. With the option off none of these keys exist.
- `cusp_proximity_apply` (the bhava chalit shift): a behind-cusp planet also counts as an occupant (level B) of the next house, so planets in its star count at level A there. Levels C/D (ownership) and the natal placement (`house`, cusp occupants) never change. Judgments, fruitful-significator selection, `find_event_periods` and the A-D tables then see it; `judge_event` / fruitful `via_A_or_B` now use all A and B houses instead of the first one (identical when the rule is off). House-wise B list now comes from the same table as the planet view, so both views agree.
- Not done / open: the 4-step view (`kp_methods`) keeps its own fixed 3d20' rule and ignores `cusp_orb_deg`; type E and six-fold use the shifted A/B lists only through `chart.sig`; no separate treatment of a planet behind a cusp whose next house is also within the orb (houses narrower than the orb); orb is the same for every planet (no planet-specific orbs); schools differ on whether a planet past a cusp changes anything.
- 37 new checks (256/256): synthetic cusps for edge cases (wrap 12->1, exact-on-cusp, inside/outside the orb, rule off), brute-force comparison of the entries against all planets and cusps of the test chart (Sun, Moon, Mars, Saturn behind cusps at 3d20'), report mode leaves A-D and verdicts identical, apply mode invariants (B/A supersets, C/D unchanged, natal placement unchanged, house and planet views agree), option validation, tool-level checks.

### Item 6 implementation notes (done 2026-11-20)
- New `kp_csl_ext.py` (pure helpers); option `KPOptions.csl_extension` = `off` (default) | `own_bhava` | `sub_chain` | `both` (aliases `ksk`, `own`, `kar`, `sub`, `all`, `none`).
- Research this session (two searches) confirmed the surrounding theory (4-step: the sub acts as a planet at step 3; EZine 128: in the six-fold table the sub is an outright significator and then cannot also be the decider) but NOT the wording of the KSK note. The rule below is therefore the backlog's reading, stated in every output; nothing was invented beyond it.
- `own_bhava`: if no planet (other than the CSL itself) stands in the star of the CSL, the CSL also signifies the bhava of the cusp it is the sub lord of. Block `csl_extension.own_bhava` lists the condition, the planets in its star and `adds` (empty when the CSL already signifies that house).
- `sub_chain` (Kar / EZine 128 style): the sub lord of the CSL is an outright significator: the house it occupies and the houses it owns are added (`csl_extension.sub_chain`). Node agents are not expanded here.
- Output: every CSL report (`get_cuspal_sublord_analysis`, `judge_event.csl`) gets a `csl_extension` block (`base_houses`, `added_houses`, `signifies_extended`, `school`). `judge_event` adds `extended_reading` (verdict with the widened houses, `changes_verdict`) beside the UNCHANGED `verdict`. A-D tables, `sub_lord_signifies`, promise table, periods and fruitful selection never use the extension.
- On the test chart the extended reading changes the verdict for several matters (marriage, property purchase, education), which is why it is a separate reading and not folded in.
- Not done / open: not applied to period search or fruitful selection; no `apply` switch like item 5 (deliberate: it would change verdicts); node CSLs do not add agent houses; the Kar interlink side (sub-sub lord of a cusp) is item 7.
- 18 new checks (274/274): mode names and aliases, validation, off = no keys, brute-force check of both readings on all 12 cusps, A-D / base houses / standard verdicts identical under every mode, `both` = union, extended reading only adds hits and flags changed verdicts, tool-level checks and a tool error for a bad mode.

### Item 7 implementation notes (done 2026-11-20)
- Attribution corrected: the Sub-Sub / Cuspal Interlinks theory is S.P. Khullar's (KCIL; books "Key to Learn Sub Sub and Cuspal Interlinks Theory", "Horary Astrology and Cuspal Interlinks"), with K. Bhaskaran's "inter-link of cuspal sub lords" beside it. The earlier backlog text said "Dr. Kar"; Kar is the retrograde research. Output says "not pure Krishnamurti".
- Sources used (search this session): Khullar: the sub-sub lord of a cusp indicates the promise and potential of that cusp, the sub lord qualifies it (e.g. Jupiter through its star lord Saturn). Bhaskaran: the CSL's star lord and sub lord are themselves CSLs of other bhavas = the inter-links (example: Venus CSL of 2nd, its star lord CSL of 9th, its sub lord CSL of 1st). Generic CIL articles: the houses the CSL signifies are the houses its cusp links with.
- New `kp_cil.py` + tool `get_cuspal_interlinks(state, cusps=None, event=None, custom_favourable=None, custom_denial=None, primary_cusp=None, levels=None)` (32 tools). Per cusp: sign / star / sub / sub-sub lord, `csl` (houses signified, `links_to_other_houses`, cusps the CSL is CSL of, its star lord and the cusps that star lord is CSL of, its sub lord and the cusps that sub lord is CSL of) and `ssl` (houses signified, same_as_csl, cusps it is CSL / SSL of). `cusp_rulership`: per planet, the cusps it rules as sign / star / sub / sub-sub lord.
- With an event (or custom houses): `event_reading` = the standard CSL verdict, plus the same favourable / denial house test applied to the SSL, and whether they agree. The SSL test is a reading of Khullar's rule, not a verdict of the tool; the standard verdict stays the CSL's.
- Not done / open: Bhaskaran's four link kinds (page not legible), the Khullar sub-sub dasha and transit techniques, the 4-step style primary / secondary split for the SSL, use of interlinks in period search; interlinks use A-D houses (levels option honoured).
- 17 new checks (291/291): lords equal the cusp rows, rulership brute force at all four levels (every cusp exactly one ruler per level), CSL / SSL houses equal the planet's signification, link lists brute-forced, event reading consistent with `judge_event`, primary cusp outside the requested subset, bad cusp, tool tests with event and custom houses.

### Item 8 implementation notes (done 2026-11-20)
- New `kp_double.py` + tool `find_double_transit(state, event=None, houses=None, primary_only=False, start=None, end=None, sign_scope="cusp", include_lord=False, limit=200)` (33 tools). Default window 3 years from `start` (today), max 30 years. Engine was found already written (unlogged earlier stop); this session audited it, fixed one bug, added tool-level tests and logged it.
- Rule: a natal house counts while BOTH transiting Jupiter and Saturn occupy or aspect a target sign of it. Aspects are sign-based from the transiting planet's own sidereal sign (7th for both; Jupiter also 5th/9th, Saturn also 3rd/10th; same `ASPECTS` table as the node-agent and type-E code). Targets: `sign_scope` `cusp` (sign of the cusp, default) | `house` (every sign the unequal Placidus house spans); `include_lord` adds the natal sign of the house lord. Houses come from `houses` (validated 1..12, sorted, de-duplicated) or from `event` (its favourable houses, or just the primary cusp with `primary_only`).
- Station handling (built on `kp_retro.find_stations`): stations of both planets are fetched for the window +- 60 days; each planet's sign path is sampled on a 5-day grid PLUS every station, so the path is monotonic between samples and a short dip back over a sign boundary after a retrograde turn is never missed; sign changes are bisected to 0.25 s. Windows are built from the pieces where both planets' relations hold; each window carries `phases` (the signs, retrograde flags and `via`: which target sign, which aspect, why) so a station that breaks a window in two is visible, `stations_within_60_days_of_window` (the "stationary transit" heuristic, reported as facts only), `starts_at_window_start` / `continues_after_window_end` flags for windows cut by the query.
- Output: `houses` (target signs, windows per house), a flat `windows` list sorted by start (with `truncated` at `limit`), `at_start` (signs, retrograde flags, houses already double transited), `stations`, `school` and `rule` strings.
- Bug fixed this session: the aspect label read "aspects (3th)" for Saturn's 3rd; added an ordinal helper (this was the one failing check). Same wording slip still exists in `kp_chart.py` and `kp_methods.py` ("aspect (3th)"), cosmetic, not touched.
- Spot check (test chart, marriage, 2026-11-20 .. 2030-11-20): Jupiter Leo, Saturn Pisces retrograde at start, Saturn direct 2026-12-11; 7th-house windows split around Saturn's 2027-08-09 retrograde / 2027-12-24 direct stations as expected.
- Not done / open: transit planets use sidereal sign aspects only (no orb / degree aspects, no star-lord or sub-lord layer: that is item 9); Rahu/Ketu not part of the rule; the Jupiter-or-Saturn single-planet windows are not listed; no combination with dasha periods (a joint "dasha + double transit" search is a possible follow-up).
- 27 new checks (318/318): aspect table incl. wrap-around, a synthetic planet that dips back over a sign boundary (caught only with the station split), brute-force comparison with raw swisseph on 395 sample dates (49 inside windows), at_start and the published Saturn station 2026-12-11, window ordering / phase tiling / flags, `primary_only`, widening by `sign_scope` / `include_lord`, input validation, and tool-level calls and tool errors.

### Item 9 implementation notes (done 2026-11-20)
- New `kp_transit.py` + tool `verify_event_day(state, when=None, event=None, custom_favourable=None, custom_denial=None, primary_cusp=None, depth=4, planets=None, extended_rps=True, levels=None)` (34 tools). `when`: date only = 12:00 local, or full datetime; default now.
- Output: `running_dasha` (chain to `depth` 1..5, each lord's signified houses, and with an event/custom houses the favourable / denial split + sub-lord check), `target_lords`, `ruling_planets_at_moment`, `transits` (default Jupiter, Saturn, Sun, Moon; any of the 9 via `planets`), `summary`, `rule`, `note`, `school`.
- Rule: a transiting planet touches a dasha lord when its SIGN lord or STAR lord is that lord (Reader 5 two-step) = `match`; sub-lord-only = `sub lord only`; else `no match`. Reported for the dasha chain and separately for the ruling planets of that moment. `planet_is_a_target_lord` is also given. `both_luminaries_touch` encodes "no event unless transiting Sun and Moon touch the wanted lords" (None when Sun or Moon is not in `planets`). Never a prediction.
- Not done / open: no window scan of days where Sun + Moon both touch (use `find_transit_triggers` for lord stretches); no matter-specific planet defaults (Mars for property, Venus for Moon dasha); sookshma-level Moon timing needs the clock time; Rahu/Ketu as transit planets use their own lords, not their agents.
- 15 new checks (333/333): transit lords and statuses vs an independent raw swisseph recomputation, summary consistency, event-house status vs `lord_status`, depth / planet list handling, validation errors, tool-level calls and errors.

### Item 10 implementation notes (done 2026-11-20)
- New `kp_longevity.py` + tool `get_longevity_facts(state, start=None, end=None, depth=3, include_periods=True, extra_houses=None, limit=100, levels=None)` (35 tools). Window default today .. +5 years. Facts only: no lifespan estimate, no death prediction, no verdict words; every output carries a note saying so.
- Output: `groups_used` (supporting 1/5/9/10, difficult 6/8/12, maraka 2/7, badhaka house, critical-period houses, classical longevity group 1/5/8/11 vs 2/7/badhaka), `lagna` kind, `ascendant_csl` and `eighth_csl` (CSL, houses signified, supporting / difficult hits, retrograde flag), `classical_group_hits` (house hits of the existing `longevity` event, no verdict), `badhaka` and `maraka` blocks (house, sign, lord, lord's house / ownership / significations), `critical_periods` (joint periods where every Dasha / Bhukti / Antara lord signifies a maraka / badhaka house, each flagged `critical_period` with the reason; `extra_houses` adds houses, e.g. 8).
- Not done / open: the supporting / difficult sets are the backlog's wording (schools differ; the existing `longevity` event uses 1/5/8/11); no ruling-planet or transit layer on the critical periods (use `verify_event_day`); no longevity "classes" (short / medium / long) by design.
- 12 new checks (345/345): groups and badhaka by lagna kind, CSL and hit sets vs direct computation, maraka / badhaka lords and significations, every period lord signifies a critical house, equality with the joint-period search, `include_periods`, `extra_houses`, note wording, validation, tool-level call and errors.

### Item 11 implementation notes (done 2026-11-20)
- New `kp_horary.py`; `judge_horary` tool gets a `horary_extras` block (additive; the judgment, verdict and other keys are unchanged) and a new parameter `asked_before` (default False). Tool count stays 35.
- `genuineness`: Lagna star lord linked to Moon sub lord (same planet, or one in the star of the other), Moon star lord signifies the matter's houses, Lagna star lord / Moon sub lord are ruling planets; each line True / False with its planets, `passed` / `total`.
- `horary_checklist`: CSL not retrograde, CSL not in a retrograde star (uses `kp_retro.is_retro`), CSL linked to a supporting (favourable) house, CSL signifies enough favourable houses (promise threshold), no denial house, CSL is a ruling planet, the CSL's own star lord is a ruling planet, Moon sub lord signifies a favourable house; `passed`, `total`, `all_passed`. No overall verdict; the standard verdict is untouched.
- `lagna_point`: which point was used (start / middle / end / actual_if_inside) with its meaning, the numbered division (start / end DMS, sign, star / sub lord), the cusp time, the actual Ascendant at the question and whether it lies inside the division, and the text of all alternatives.
- `reminders`: first-sincere-asking-only, spontaneous number, one question / one chart, time sensitivity. `first_asking_only` is added only when `asked_before=true`.
- Fixed during testing: the old judgment field `csl_star_lord_is_ruling_planet` is the CUSP's star lord, not the CSL's star lord (misleading name, left unchanged for compatibility); the checklist computes the CSL's own star lord separately.
- Not done / open: number spontaneity cannot be computed; no automatic comparison of all `lagna_point` options (re-run with another point); checklist weighting differs by school (no threshold applied); judgment fields keep their old names.
- 7 new checks (352/352): link / RP / Moon-star lines vs direct computation, checklist lines vs direct computation, lagna_point explanation, reminders and `first_asking_only`, verdict unchanged, tool-level calls.

### Item 12 implementation notes (done 2026-11-20, except 12b)
- New `kp_rectify.py` (pure helpers: `lagna_moon_link`, `event_weight`, constants); `kp_moment.rectify` and the tool `rectify_birth_time` extended, all additive (old keys kept; tool count stays 35). New tool parameter `moon_link` ('report' default | 'require'); events may carry `weight`.
- (a) Lagna-Moon link: the Lagna star / sub lord connect with the Moon star / sub lord = same planet, same sign (association) or sign aspect (7th for all, Mars 4/8, Jupiter 5/9, Saturn 3/10; nodes only associate). Every range carries `lagna_moon_link` (`linked`, lords, `shared_planets`, `links`); the link is part of the range grouping. 'report' only shows it; 'require' ranks only linked ranges (`filtered_out_ranges`, `moon_link.ranges_filtered_out`; if no range is linked the filter is not applied and `no_range_linked_so_filter_not_applied` says so).
- (c) Weights: score per event = (CSL verdict points + 1 per favourable running lord) x `weight` (0 < w <= 10, default 1; validated); `max_score_possible` = 6 x sum of weights; `weights_used`; `anchor_events` lists the usual anchors (accident, marriage, child birth, service / promotion, foreign travel, property purchase) with a note that a parent's death has no event key (use `custom_favourable`).
- (d) Sensitivity: every best range gets `lagna_sub_lord_sensitivity` (Lagna sub lord at the middle of the range, minutes back / forward until it changes, stretch minutes, `cut_by_scan_window` when the stretch reaches the +-60 minute scan limit).
- (b) NOT done: the "Rule of Origin" and the Kanak Bosmia 2-minute method are named in KP e-zines but their rules were not verified, so nothing was coded; `caution` says so. Needs research first.
- Not done / open: link rule is one reading of "connect" (schools differ); weights are the caller's choice (no defaults per event type); the link is evaluated at each trial time, the sensitivity only for best ranges; cusp sub lords other than the Lagna are not part of the sensitivity.
- 15 new checks (367/367): link consistency and per-range link vs recomputation, weights scale scores / maximum, weight validation, moon_link modes and validation, sensitivity vs the Ascendant at the range middle, anchors / caution, tool-level call and errors.

### Item 13 implementation notes (done 2026-11-20)
- `kp_twins.compare_charts` + tool `compare_births(a, b)` (36 tools). Options of `a` apply to both. Output: `minutes_apart`, `same_place`, `lagna`, 12 `cusps` rows (sign / star / sub / sub-sub lords of each side, `differs_at`, `sub_lord_differs`, `moved_degrees`), `summary` (cusps differing by sub / star / sign, planets that differ), `planets` (only those differing, incl. house change), `dasha_balance` (both, `same_start_lord`, `balance_difference_days`), `events` (only where the CSL or verdict differs), `rule`, `note`. Facts only.
- Spot check (test chart, 10:30 vs 10:36): 10 of 12 cusp sub lords differ, Lagna moved 1.335 deg (Jupiter/Rahu -> Saturn/Saturn), no planet differs, 28 events change CSL or verdict.
- Not done / open: no search for the smallest time gap that flips a given cusp sub lord (use `get_lagna_changes` for the Lagna); no twin dasha-period divergence table.
- 10 new checks (377/377): identical births, cusp rows vs direct recomputation, summary lists, Lagna motion, events filter, dasha balance sign, place flag, option sharing, tool error.

### Item 15 implementation notes (done 2026-11-20)
- New `kp_muhurta.py` + tool `find_muhurta_windows(moment, event|custom_favourable, end=+24h, custom_denial, primary_cusp, step_minutes=10, csl_min=weak, require_lagna_sub=True, min_rp_support=2, limit=50, levels)` (37 tools). `moment` = start, date/time/place; options of the moment state apply.
- Per sample the chart is cast and three tests run: Lagna sub lord signifies a favourable house and no denial house; the matter's CSL verdict rank >= `csl_min` (none | weak = weakly_promised / promised_with_obstacles / strongly_promised | strong = strongly_promised only); at least `min_rp_support` distinct ruling planets (day lord, Moon sign / star / sub, Lagna sign / star / sub) signify a favourable house without a denial house. Consecutive passing samples = a window (`from`, `until_about`, Lagna sub lords and CSL verdicts seen, RPs at start, `continues_after_window_end`). Max 4000 samples; `truncated` flag on `limit`.
- Not done / open: window edges are accurate to the step only (confirm with `get_lagna_changes`); no panchang / tithi / yoga / Moon-nakshatra avoidance rules (not KP; not an auspiciousness verdict); no matching of the moment against a person's natal chart (e.g. the native's own RPs); day lord is looked up per hour of the clock, sunrise edge is approximate within the hour cache.
- 13 new checks (390/390): sample count, passing = sum of windows, every window start recomputed independently and the sample after the end fails, stricter / looser options monotone, limit, custom houses = event table, validation errors.

### Item 16 implementation notes (done 2026-11-20)
- With `include_outer` on: `get_kp_chart` gets an `outer_planets` block (house, sign, lords, retrograde, rule text), cusp rows and house-significator rows get `outer_occupants`. Planet rows already listed them. Occupants only: A-D tables, planet significations, judgments, dasha are provably identical with the option on or off (tested). Off (default): none of the new keys exist.
- Not done / open: no star-lord role (they own no star); not used as aspecting bodies in type E / node agents; no sub-level use.
- 6 new checks (396/396).

### Item 27 implementation notes (done 2026-11-20)
- New `kp_report.py` + tool `get_kp_report(state, on_date=None)` (38 tools): returns `markdown` (planets with sign / star / sub / sub-sub / house / houses signified, cusps with CSL and the houses it signifies (the cuspal-links table), significators by house A-D, balance at birth, running ladder to sookshma, ruling planets on `on_date` at the birth place, Fortuna). No verdicts. `get_kp_summary` is unchanged.
- Not done / open: the Python-side promise table is not in the report (by design, facts only); no PDF / file output (caller saves the text); outer planets not listed.
- 6 new checks (402/402): sections, header, rows vs chart, CSL column, ladder, no verdict words.

### Item 26 implementation notes (done 2026-11-20)
- New `kp_workflow.py` + tool `kp_judgment_workflow(state, event|custom houses, primary_cusp, start, end, when, depth=3, limit=10, levels)` (39 tools). One calc context; steps: 1 matter / cusp (info), 2 CSL (info, with retrograde check), 3 promise (pass = strongly_promised; partial = weakly_promised / promised_with_obstacles; fail = doubtful / not_promised), 4 timing (pass if any joint period in the window), 5 transit check on `when` (pass when both transiting Sun and Moon touch the dasha lords; n.a. without `when`), 6 ruling planets at `when` (else at `start`): pass = at least two signify favourable houses and no denial house, partial = one. Each step carries `rule`, houses and data of the underlying tool; `status_counts`; NO overall verdict.
- Not done / open: no fruitful-significator step; step 5 uses the Sun / Moon rule only (planets list is the default); step 6 RPs use the birth place; thresholds are this tool's reading, stated in the `rule` text.
- 12 new checks (414/414): step order, equality with judge_event / find_event_periods / verify_event_day / get_ruling_planets, status mapping, custom houses, n.a. without `when`, validation errors.

### Item 19 implementation notes (done 2026-11-20)
- `kp_chart.node_check(chart)`; `get_kp_chart(detail="full")` adds `node_check`: the node type in use vs the other type at the chart time, per node the sign / star / sub lords of both, `differs_at`, `sub_lord_differs`, `difference_deg`, `any_sub_lord_differs`. Standard / basic output unchanged (key absent). Fact only; the chart itself never switches type.
- Not done / open: not in `get_kp_summary` or the report; no flag in period tables; only the natal chart time.
- 5 new checks (419/419): absent unless full, mean/true symmetry, Ketu = Rahu + 180, flag consistency.

### B. New KP techniques found in research (priority 2, in rough value order)
5. **[DONE 2026-11-20, see "Item 5 implementation notes"] Cusp proximity / "bhava chalit" rule** (R17): a planet close behind a cusp (within about 3d20') also signifies the cusp in front of it; planets ON a cusp are strong significators of it. Option `cusp_orb_deg` (default off), adds a `near_cusp` list to the chart and to significators. Directly affects the planet-in-house table, so test on edge cases.
6. **[DONE 2026-11-20, see "Item 6 implementation notes"] Cusp sub lord as significator of its own bhava when no planet sits in its star** (KSK note) and the Kar "sub as significator" extension, flagged as not-pure-KP.
7. **[DONE 2026-11-20, see "Item 7 implementation notes"] Cuspal Interlinks (CIL; the school is S.P. Khullar's KCIL and K. Bhaskaran's inter-links, NOT Dr. Kar as first written here)**: sub-sub lord of a cusp as the promise indicator; links between cusp sub lords (e.g. 2nd CSL linked to 11th). Tool `get_cuspal_interlinks(state)` listing each cusp's sign/star/sub/sub-sub lord with the houses each signifies, flagged "CIL school, not pure KP".
8. **[DONE 2026-11-20, see "Item 8 implementation notes"] Double transit (Jupiter + Saturn)** (R18): given event houses, list windows when Jupiter and Saturn both aspect/occupy the sign of the relevant cusps (K.N. Rao / Parashari technique often used beside KP). `find_double_transit(houses|event, start, end)`, with retrograde station handling (the aspect can shift for months when Saturn retrogrades). Plus KP-style **stationary transit** note: events cluster within about 2 months of a Jupiter/Saturn station.
9. **[DONE 2026-11-20, see "Item 9 implementation notes"] KP transit rule set** (R9 extension): for Jupiter, Saturn, Sun, Moon check that the transiting planet is in the sign / star / sub of the dasha-bhukti-antara lords; "no event unless transiting Sun and Moon occupy the wanted positions"; transit of DBA lords together with the luminaries; event-day verification tool `verify_event_day(date)` = running dasha chain + transiting Sun/Moon/Jupiter lords + ruling planets of that day, each marked match / no match.
10. **[DONE 2026-11-20, see "Item 10 implementation notes"] Longevity / maraka / badhaka module**: CSL of the Ascendant and of the 8th signifying 6/8/12 vs 1/5/9/10; badhaka and maraka (2, 7) joint periods flagged "critical period". Facts only, with wording that avoids predicting death; classical-rule listing, caller decides how to present.
11. **[DONE 2026-11-20, see "Item 11 implementation notes"] Horary extras** (R19): genuineness check (Lagna star lord to Moon sub lord link; Moon star lord signifying the matter's houses), "first sincere asking only" reminder, the CSL-not-retrograde / not-in-retrograde-star / linked-to-supporting-house checklist as a structured `horary_checklist`, and a `lagna_point` explanation. Number-must-be-spontaneous is a user-process note, not computable.
12. **[DONE 2026-11-20 for (a), (c), (d); (b) NOT done, see "Item 12 implementation notes"] Birth-time rectification upgrades** (R20): (a) Lagna star/sub lord must connect with the Moon's star/sub lord (association/aspect) as a pre-filter; (b) "Rule of Origin" and the Kanak Bosmia 2-minute method named in KP e-zines (details unknown, research before coding); (c) several-event scoring weights, accident/marriage/child-birth/parent-death events as the standard anchors; (d) report the sensitivity (how many minutes until the Lagna sub lord changes) with each candidate.
13. **[DONE 2026-11-20, see "Item 13 implementation notes"] Twin / close-birth comparison**: `compare_births(a, b)` listing which cusp sub lords differ between two birth times (KP's twin-divergence demonstration; sub-lord change can happen in about 2m36s at some cusps).
14. **[DONE 2026-11-21, reduced scope] Medical astrology (KP)**: 6th CSL + disease house groups (1, 6, 8, 12, badhaka/maraka), planet-to-body-part table (Reader 6 / Medical Astrology). Needs a vetted table before coding; facts only and a prominent not-medical-advice note.
15. **[DONE 2026-11-20, see "Item 15 implementation notes"] Electional (muhurta) helper**: scan a date range for moments whose Lagna sub lord / cusp sub lords support a chosen matter's house group and whose ruling planets match (we already have `ruling_planets_timeline` and `lagna_changes` as building blocks). Tool `find_muhurta_windows(event, place, start, end)`.
16. **[DONE 2026-11-20, see "Item 16 implementation notes"] Uranus / Neptune / Pluto**: they are used in KP (12 bodies; they own no sign or star, so they sit only as occupants/aspectors). Add them as occupants in significators when `include_outer` is on (default off), and in `get_kp_chart` output.
17. **Market / financial KP and mundane use**: Gondhalekar-style stock-market notes and "KP graphs" exist but are speculative and not rule-defined enough. Park as research-only; if ever built, a facts tool (transit lords of an index chart) with an explicit "not financial advice" note.

### B2. Added after research round 4 (2026-11-20; ideas, each still needs a vetted rule source before coding)
25. **[DONE 2026-11-21] KP annual horoscope (Varshaphal, KP flavour)**: solar-return chart (Sun back to natal sidereal longitude, binary search on JD) cast for the NEW place of residence, not the birth place (per KP Forum talk, R. Nimje), then read with KP cusp sub lords and compared with the natal chart; optional Tajik / Mudda dasha for the year (starts from the Moon star at the return, interpreted by KP significators); optional Muntha. Tool `get_kp_annual_chart(state, year, place)`. Tajika yogas are NOT KP, so leave them out or flag them as non-KP.
26. **[DONE 2026-11-20, see "Item 26 implementation notes"] Guided KP judgment workflow** (the "7 fundamental steps" teaching sequence: pick the matter -> cusp -> CSL -> promise -> DBA timing -> transit -> ruling planets verification): one tool `kp_judgment_workflow(state, event, moment?)` that runs the existing tools in that order and returns one structured, auditable result (each step: rule used, houses used, pass / fail / n.a.). Composition only, no new astrology.
27. **[DONE 2026-11-20, see "Item 27 implementation notes"] One-page condensed report + enhanced cuspal-links table** (what KP desktop software prints): planets and cusps with sign / star / sub / sub-sub, 4-level significator table, running dasha ladder, RPs now, CSL link table (each cusp's CSL and the houses it signifies). Extends `get_kp_summary`; output as text/markdown so it can be saved or printed.
28. **Scope notes (do NOT build, recorded so they are not re-researched)**: Nadi-style 150-per-sign Nadi astrology, Bhrigu Nandi Nadi, Jaimini chara dasha, Ashtottari / Yogini dasa, Ashtakavarga and Shadbala are separate schools, not KP. KP uses Vimshottari only; mixing them would break the "one rule, stated" design rule. Revisit only if the user asks.

### C. Quality, docs and plumbing (priority 3)
18. **Golden charts from published KP examples** (Reader 1-6 / EZine charts with listed cusps, sub lords and dasha) as regression tests; we only cross-check against raw swisseph and six table longitudes so far.
19. **[DONE 2026-11-20, see "Item 19 implementation notes"] True-node warning**: flag when a node's sub lord differs between mean and true node at the chart time.
20. **[DONE 2026-11-21 as audit; sources conflict, see log] Ayanamsa table check**: compare `KP_NEW` / `KP_OLD` output with a published KP ephemeris for several dates (the nutation convention, see session log, explains a ~12" gap against some printed tables).
21. **[DONE 2026-11-21] Place helper**: tool that returns latitude / longitude / timezone for a place name (offline table or geopy already in requirements) so agents do not invent coordinates; historical timezone/DST handling check for old birth dates.
22. **Orchestrator agent plugin `agents/kp.py`** (not asked yet): description for the router, birth-details memory, tool-grounded answers like the Dasha agent. Needs the user's go-ahead.
23. **[DONE 2026-11-21] User docs**: KP section in Readme with tool list by group, option glossary (ayanamsa, node type, significator method, retrograde rule, orb), and the school-difference table from this file.
24. **[DONE 2026-11-21, audit] Performance guard**: `rectify` and boundary scans are fast (seconds), but add result-size limits and a `truncated` flag everywhere a list can grow.

Rules (from the project): install NOTHING (only edit `requirements.txt`); never print secrets; run with `.venv/bin/python`;
long jobs go through `start_process` + log polling; no `__init__.py` in `mcp/`; avoid recursive grep on the project.

## Design (decided 2026-11-20)
**Backend:** `pyswisseph` only (Moshier fallback is what this PC uses: no ephemeris files; ~1" planets, ~3" Moon: fine for KP). No PyJHora.
**Sidereal:** `swe.set_sid_mode(mode)` + `FLG_SIDEREAL`; `houses_ex(..., b'P', FLG_SIDEREAL)` = tropical Placidus minus ayanamsa (verified). Ayanamsa options:
`KP_NEW` (default, swe 45 VP291, matches published KPNA), `KP_OLD` (swe 5), `LAHIRI` (1), `RAMAN` (3).
Sunrise for the Hindu weekday: `swe.rise_trans(jd_ut, SUN, CALC_RISE, (lon, lat, 0), 0, 0, FLG_SWIEPH)` (verified, Mumbai 6:05 in May).
Horary Lagna: solve the time (same place, +-1 sidereal day) where the sidereal Placidus Ascendant equals the chosen point of the numbered sub; cusps from that time, planets from the question moment.
Nodes: mean by default (`node_type=mean|true`), Ketu = Rahu + 180. Dasha year = 365.25 days (option).
Planet ids everywhere: names (Sun..Ketu); internal swisseph ids Sun0 Moon1 Mercury2 Venus3 Mars4 Jupiter5 Saturn6 (own order used in code, API uses names).

**Files (mcp/kp/, no __init__.py):**
- `state.py`: `KPState` (birth: same fields as dasha/transit + `options`), `MomentState` (date/time/place for horary, RPs, transits), `KPOptions`.
- `kp_tables.py`: PURE (no swisseph): signs, nakshatras, Vimshottari order/years, 243 subs -> 249 numbered table (integer arc-seconds), `kp_lords(lon)` (sign/star/sub/sub-sub/sookshma), house-group event table, house themes, badhaka.
- `kp_core.py`: swisseph backend (lock, ayanamsa, jd, planet longitudes/speeds, cusps, sunrise, Hindu weekday), `set_backend` for tests.
- `kp_chart.py`: chart assembly, planet & house significators (A/B/C/D + agents), cuspal sub-lords, event judgment (promise / denial), fortuna.
- `kp_dasha.py`: Vimshottari (KP year), periods to sookshma, running periods, DBA significations, joint-period search.
- `kp_moment.py`: ruling planets, horary, transit snapshot, boundary (sign/star/sub) change search, ruling-planet timeline, Sun/Moon trigger search, rectification scan.
- `kp_server.py` + `main.py` (port 8002, `KP_MCP_HOST/PORT`, stdio or `--http`), `test/test_kp.py` (+ smoke scripts).

**Planned tools (target ~30):** reference (get_kp_sub_table, lookup_longitude, get_ayanamsa, list_event_types, get_house_groups), chart (get_kp_chart, get_cusps, get_planets, get_significators, get_planet_significations, get_cuspal_sublord_analysis, judge_event, get_fortuna), dasha (get_kp_dasha, get_current_kp_dasha, get_dasha_significations, find_event_periods, analyze_joint_period), moment (get_ruling_planets, get_ruling_planets_timeline, cast_horary_chart, judge_horary, get_transit_snapshot, find_boundary_changes, find_transit_triggers, rectify_birth_time, get_kp_summary).

**Conventions:** tools state facts and classical KP rules only; they never predict outcomes; every judgment returns the houses/levels it used so the caller can audit it.

## Session log (newest first)
- **2026-11-21 (item 14 of backlog): health facts DONE (reduced scope).** `kp_medical.py`, tool `get_health_facts` (43 tools), 8 new checks (447/447 pass). Done: 6th CSL with disease houses 6/8/12, 1, recovery 5/11, nature hints (Saturn chronic, Mercury complications, Mars sudden; from a Reader 6 summary page), traditional sign-body table (flagged general Vedic), Mars significations, 8th / 12th CSL, disease and recovery joint periods. NOT done: planet-to-body-part table and star / sub disease tables (no vetted source); no surgery rule (only a blog source). Prominent not-medical-advice text in output. Next: 18 (golden charts, needs published chart data), 22 (needs go-ahead).
- **2026-11-21 (item 25 of backlog): KP annual chart DONE.** `kp_annual.py`, tool `get_kp_annual_chart(state, year, place=None, detail)` (42 tools), 8 new checks (439/439 pass). Return = sidereal Sun back to natal longitude (bisection, +-4 days around birthday), cast for `place` (default birth place); output: return moment, chart, promise table, cusp / planet comparison with natal, natal dasha ladder at the return. Tajika yogas / Muntha / Mudda dasha NOT done (not KP, by design). Open: no KP-specific reading rules beyond facts; no dasha started from the return Moon. Next: item 14 (needs vetted table; research), 22 (needs go-ahead), 18 golden charts, then Readme count update (42 tools / 439 checks).
- **2026-11-21 (items 24, 20): audit only, no code change.** 24: every list tool already has `limit` + `truncated` (periods, double transit, longevity, muhurta, boundary / trigger scans) or a hard error (`get_kp_dasha` MAX_ROWS, rectify / timeline MAX_CANDIDATES, muhurta MAX_STEPS) = DONE. 20: web check found NO reliable published KP table: one source gives KP Old (Krishnamurti table) 23:45:56 at year 2000 (swe mode 5 gives 23:45:36.9 at J2000, ~19" apart, flat-year table vs continuous), a JHora page claims KP New ~ Lahiri +5" while others say Lahiri minus ~4m54s (our mode 45 = Lahiri - 4m36s); sources contradict, so the ayanamsa values are left as swisseph provides and the conflict stays documented (R7). Next: item 25 (annual chart).
- **2026-11-21 (item 23): Readme KP section DONE** (tool groups, option glossary, counts 41 tools / 431 checks). Next: item 24 (size-limit audit), 25 (annual chart), 20.
- **2026-11-21 (item 21 of backlog): place helper DONE.** `kp_place.py`, tools `get_timezone_info` (offline, DST / ambiguous / non-existent flags) and `lookup_place` (Nominatim via geopy, needs internet, clean tool error otherwise) (41 tools), 12 new checks (431/431 pass). Next: item 25 (annual chart), 24 (size-limit audit), 20 (ayanamsa check), 23 (docs).
- **2026-11-20 (item 19 of backlog): true-node warning DONE.** `kp_chart.node_check`, `get_kp_chart(detail='full')` adds `node_check`, 5 new checks (419/419 pass). Next: item 21 (place helper), 25 (annual chart), 23 (docs), 24 (size-limit audit), 20 (ayanamsa table check).
- **2026-11-20 (item 26 of backlog): guided workflow DONE.** `kp_workflow.py`, tool `kp_judgment_workflow` (39 tools), 12 new checks (414/414 pass). Next: item 19 (true-node warning), 21 (place helper), 25 (annual chart), 23 (docs), 24 (size limits audit).
- **2026-11-20 (item 27 of backlog): condensed markdown report DONE.** `kp_report.py`, tool `get_kp_report` (38 tools), 6 new checks (402/402 pass). Next: item 26 (workflow), 19 (true-node warning), 21 (place helper), 25 (annual chart), 23 (docs).
- **2026-11-20 (item 16 of backlog): outer planets as occupants DONE.** `kp_chart.outer_occupants` / `outer_block`, 6 new checks (396/396 pass). Next: item 27 (condensed report), 26 (workflow), 25 (annual chart), 21 (place helper), 14 needs a vetted table.
- **2026-11-20 (item 15 of backlog): muhurta windows DONE.** `kp_muhurta.py`, tool `find_muhurta_windows` (37 tools), 13 new checks (390/390 pass). Next: item 16 (outer planets as occupants), then 14 (needs vetted table), 25-27.
- **2026-11-20 (item 13 of backlog): twin / close-birth comparison DONE.** `kp_twins.py` (found written, unlogged), tool `compare_births` (36 tools), 10 new checks (377/377 pass). Next: item 15 (muhurta), then 16, 14 (needs vetted table).
- **2026-11-20 (item 12 of backlog): rectification upgrades DONE (a, c, d; b not verified).** `kp_rectify.py`, `rectify` / `rectify_birth_time` extended (`moon_link`, event `weight`, Lagna sub-lord sensitivity), 15 new checks (367/367 pass). Next: item 13 (twin / close-birth comparison).
- **2026-11-20 (item 11 of backlog): horary extras DONE.** `kp_horary.py`, `judge_horary` gets `horary_extras` + `asked_before` (35 tools), 7 new checks (352/352 pass). Next: item 12 (rectification upgrades).
- **2026-11-20 (item 10 of backlog): longevity / maraka / badhaka facts DONE.** `kp_longevity.py`, tool `get_longevity_facts` (35 tools), 12 new checks (345/345 pass). Next: item 11 (horary extras).
- **2026-11-20 (item 9 of backlog): KP transit rule set DONE.** `kp_transit.py`, tool `verify_event_day` (34 tools), 15 new checks (333/333 pass). Next: item 10 (longevity / maraka / badhaka module).
- **2026-11-20 (item 8 of backlog): double transit DONE.** `kp_double.py` (found written but unlogged), tool `find_double_transit` (33 tools), one label bug fixed (ordinal "3rd"), tool-level tests added, 27 new checks (318/318 pass). Next: item 9 (KP transit rule set, `verify_event_day`).
- **2026-11-20 (item 7 of backlog): cuspal interlinks DONE.** `kp_cil.py`, tool `get_cuspal_interlinks` (32 tools), 17 new checks (291/291 pass). Attribution fixed (Khullar / Bhaskaran, not Kar). Next: item 8 (double transit Jupiter + Saturn, `find_double_transit`, with station handling; uses `find_stations`).
- **2026-11-20 (item 6 of backlog): CSL extensions DONE.** `kp_csl_ext.py`, option `csl_extension` (default off), `csl_extension` block + `extended_reading`, 18 new checks (274/274 pass). Rule wording of the KSK note could not be verified; flagged in output. Next: item 7 (cuspal interlinks, sub-sub lord of a cusp; tool `get_cuspal_interlinks`).
- **2026-11-20 (item 5 of backlog): cusp proximity / bhava chalit DONE.** `kp_cusp.py`, options `cusp_orb_deg` + `cusp_proximity_apply` (both off by default), `near_cusp` blocks in chart and significators, 37 new checks (256/256 pass). Next: item 6 (cusp sub lord as significator of its own bhava when no planet is in its star; Kar sub-as-significator extension, flagged not-pure-KP).
- **2026-11-20 (item 4 of backlog): retrograde rules DONE.** `kp_retro.py`, option `retrograde_rule`, `find_stations` tool, flags in CSL check and periods, 22 new checks (219/219 pass). Section A of the backlog (items 1-4) is complete. Next: section B, item 5 (cusp proximity / bhava chalit rule).
- **2026-11-20 (item 3 of backlog): fruitful-significator filtering DONE.** `kp_fruitful.py`, tool `get_fruitful_significators`, `find_event_periods(fruitful_method, moment)`, `find_joint_periods(allowed)`, 17 new checks (197/197 pass). Next: item 4 (retrograde rules).
- **2026-11-20 (item 2 of backlog): type-E significators DONE.** `kp_methods.type_e`, options `include_type_e` / `type_e_basis` / `type_e_orb_deg`, `get_significators(include_type_e)`, 13 new checks (180/180 pass). Next: item 3 (fruitful-significator filtering).
- **2026-11-20 (item 1 of backlog): 4-step theory + 6-fold DONE.** `kp_methods.py` added, `state.py` option `significator_method`, `kp_server.py` (`get_significators(method)`, `get_planet_significations(include_sub_level)`), 19 new checks (167/167 pass). Details in "Item 1 implementation notes". Next: item 2 (type-E significators).
- **2026-11-20 (same resumed session): `kp_moment.py`, `kp_server.py`, `main.py`, tests DONE.**
  - `kp_moment.py`: `ruling_planets` (day lord sunrise-to-sunrise, Moon/Lagna sign-star-sub lords, counts, star/sub-of-other-RP flags, node representatives, optional natal overlay), `ruling_planets_timeline`, `cast_horary` (+ `judge_horary`: CSL, RP check, Moon sub lord check), `transit_snapshot` (natal house + natal significations of star/sub lords), `planet_segments` / `find_boundary_changes` (sign/star/sub/sub_sub, bisected to ~0.25 s), `find_transit_triggers`, `lagna_changes`, `rectify` (scores CSL verdict + running Dasha/Bhukti/Antara lords on each event date over +-N min trial times, grouped into equal-outcome ranges).
  - `kp_server.py`: 29 tools (reference 6, natal chart 9, dasha 5, moment/search 9). Engine errors -> clean tool errors. `main.py` port 8002 (`KP_MCP_HOST/PORT`).
  - `kp_core.py`: added `ayanamsa_values`, `julday_utc`. `kp_dasha.py`: removed two unused methods (`running`, `iter_level`; `running_ladder` is the real one).
  - **Finding (nutation):** swisseph's sidereal mode (planets AND `houses_ex(..., FLG_SIDEREAL)`) subtracts the ayanamsa WITH nutation from the true-equinox tropical value (`get_ayanamsa_ex_ut`, 23.649065 deg at 1990-05-15) - i.e. it works in the mean frame; `get_ayanamsa_ut` (23.645812, no nutation) is the value we DISPLAY and that matches the published KP/Lahiri numbers. Difference 11.7". Verified exact (0.0") for Moon and Ascendant. Tests use the `_ex_ut` value as the independent reference. KP sub boundaries are arc-minute scale, so this convention is fine, but it is why a hand check against a printed table can differ by ~12" if the printed table includes nutation.
  - Test notes: textbook longitudes tested at +0.5" (boundaries are inclusive at the start of a sub). All 249 horary numbers cast OK in ~1.5 s. Real stdio client run and `--http` start verified (server stopped afterwards).
  - Event table has **33** event types (earlier note said 34).
  - Machine note: tools run on this PC with `.venv/bin/python` (Python 3.12); nothing was installed.
- **2026-11-20 (resumed after an unlogged stop): audit of the code on disk.** Found already written but not logged: `state.py` (KPState/MomentState/KPOptions), `kp_core.py` (swisseph backend: Ctx, calc, cusps with Porphyry fallback, solve_time_for_ascendant, day_lord/sunrise), `kp_chart.py` (chart_at, significators A-D, node agents, CSL report, judge_event, promise_table, fortuna), `kp_dasha.py` (Dasha, periods_table, running_ladder, find_joint_periods), `test/smoke_chart.py`. Ran smoke_chart (Mumbai 1990-05-15 10:30): OK, chart/cusps/significators/judge_event/fortuna/agents all print sensibly. NOT yet written: `kp_moment.py`, `kp_server.py`, `main.py`, `test/test_kp.py`.
- **2026-11-20: `kp_tables.py` DONE + verified.** 243 subs -> 249 numbered divisions (6 split by a sign boundary, 12 rows); the 6 textbook longitudes from the research (1d23'55" Mars/Ketu/Venus, 33d46'54" Venus/Sun/Saturn, 59d19'19" Venus/Mars/Saturn, 83d02'49" Mercury/Jupiter/Saturn, 108d59'58" Moon/Mercury/Ketu, 141d16'58" Sun/Venus/Jupiter) all match; sub spans match the published table (Ketu 0-46-40 ... Mercury 1-53-20); every one of the 249 rows agrees with `kp_lords`. Aries has 22 divisions, Taurus 19. 33 event types encoded.
- **2026-11-20: research round 2.** Searched KP dasha timing, transits, retrograde/combust, nodes (mean vs true), rectification, KP Readers contents, Fortuna. Notes R8-R13 below; open items trimmed.
- **2026-11-20: research round 1.** Searched significators (A/B/C/D), ayanamsa, ruling planets, horary, house groups. Probed this machine's
  Swiss Ephemeris (see "Machine probe"). Existing `mcp/transit` already has `get_kp_transit`; this server must go far beyond that.
- **2026-11-20: started.** Created this file. `.venv` has pyswisseph 2.10.3.2, mcp 1.30.0, PyJHora 5.0.0.

## Phases
| # | Phase | Status |
|---|-------|--------|
| 1 | Deep research of KP (rules, tables, ayanamsa, house groups, horary, dasha, ruling planets, rectification, transit) | done (rounds 1-2, notes R1-R14) |
| 2 | Design: tool list, files, test plan | done |
| 3 | `state.py` + engine modules | done (`kp_tables`, `state`, `kp_core`, `kp_chart`, `kp_dasha`, `kp_moment`) |
| 4 | `kp_server.py` (MCP tools) + `main.py` (port 8002) | done (29 tools) |
| 5 | Tests (real swisseph, cross-checks against raw swisseph + textbook values) | done (291/291) |
| 6 | Coverage audit + final notes + requirements.txt + Readme | in progress: requirements/Readme done; gaps listed above |

## Machine probe (this PC, swisseph 2.10.03, run 2026-11-20)
| Mode | swe id | J2000 | 2026-11-20 |
|---|---|---|---|
| Lahiri | 1 | 23d51m25.5s | 24d13m57.6s |
| Krishnamurti (KP old-style, smooth) | 5 | 23d45m36.9s | 24d08m09.0s |
| Krishnamurti VP291 (matches published "KP New" ~23:47 at 2000) | 45 | 23d46m49.3s | 24d09m21.4s |
| Raman | 3 | 22d24m38.8s | 22d47m11.0s |
- Lahiri - KP(5) = 0.0968 deg (5m49s), constant. Lahiri - VP291 = 0.0767 deg (4m36s). Published "KP New = Lahiri - 4m54s" is within ~20 s of VP291.
- `swe.houses_ex(jd, lat, lon, b'P', FLG_SIDEREAL)` works at Mumbai; **raises at lat 70** (Placidus undefined near polar circle) -> need fallback + clear error.
- MEAN_NODE = 10, TRUE_NODE = 11.

## Research notes

### R1. Core structure (Sign / Star / Sub / Sub-sub)
- 27 nakshatras x 9 subs = 243; **249** numbered subs because 6 of the 12 sign boundaries fall inside a sub and split it (same sub lord, new sign).
- Sub span = (lord's Vimshottari years / 120) x 13d20m (800'). Order starts at the star lord and cycles Vimshottari order
  (Ketu 7, Venus 20, Sun 6, Moon 10, Mars 7, Rahu 18, Jupiter 16, Saturn 19, Mercury 17 = 120). Sub-sub divides the sub the same way.
- Sub spans (deg-min-sec): Ketu 0-46-40, Venus 2-13-20, Sun 0-40-00, Moon 1-06-40, Mars 0-46-40, Rahu 2-00-00, Jupiter 1-46-40, Saturn 2-06-40, Mercury 1-53-20; total 13-20-00.
- Planet = source, star = decides what it gives, **sub = decides whether/how it gives** ("sub is the deciding factor").
- KP uses Placidus cusps (house = cusp to next cusp, unequal), sidereal with KP ayanamsa, Vimshottari dasha with 365.25-day year.

### R2. Significators
- Four classical levels (strongest -> weakest): **A** planets in star of an occupant of the house; **B** occupants; **C** planets in star of the house owner (sign lord of the cusp); **D** the owner.
- Extra "type E": planets conjunct/aspecting the significators (Krishnamurti's later addition; optional, flagged as secondary).
- A planet in its own star with no planet in its star = "self-strength"; if house empty, A and B blank. Planet with no planet in its star is stronger.
- Rahu/Ketu: act as **agents** - give results of (in ascending strength) conjunction/aspect, sign lord, star lord; strong as star lords; as planets they are strong in a house only if nobody tenants their star.
- Planet signification = houses of (its star lord occupant/owner) first, then its own occupation/ownership. 6-fold variant: planet in star of A and sub of C (sub-level significations) is used in Reader 6 discussions.
- 4-step theory (Shri Bhatt): planet -> star lord -> sub lord -> sub-lord's star; a rival school; implement as optional "4-step" view later.

### R3. Cuspal sub-lord (CSL) judgment (the "promise" test)
- Event promised if CSL of the relevant cusp signifies the supporting houses and NOT the denial houses; the dasha/bhukti/antara lords (all signifying the event houses) give timing; transit confirms; ruling planets verify.
- Rule: CSL's star lord is the one whose houses count first, then the CSL's own houses, then its sub lord's chain.

### R4. House groups (promise / fulfilment) - the table to code
| Matter | Favourable | Denial / negative |
|---|---|---|
| Marriage | 2, 7, 11 (engagement 3, 9, 11) | 1, 6, 10 (separation also 12) |
| Divorce/separation | 1, 6, 10 (12th from 2, 7, 11) | |
| Child birth | 2, 5, 11 | 1, 4, 10 |
| Education | 4, 9, 11 (higher: 9, 11 and 4) | 3, 8 (completion/denial), 12 |
| Service/job | 2, 6, 10 (also 11) | 1, 5, 9 (change of job), 12 |
| Business | 2, 7, 10, 11 | 6 |
| Gain of money | 2, 6, 11 | 5, 8, 12 |
| Speculation gain | 2, 5, 6, 11 | 8, 12 |
| Promotion | 2, 6, 10, 11 | 5, 9, 12 |
| Foreign travel | 3, 9, 12 | 2, 4, 11 |
| Return home | 3, 9, 11 | |
| Property purchase | 4, 11, 12 | 3, 5, 10 (sale: 3, 5, 10) |
| Cure of disease | 1, 5, 11 | 6, 8, 12 |
| Proneness to disease | 1, 6, 8, 12 | |
| Longevity | 1, 5, 8, 11 | short life: 2, 7, 8, badhaka |
| Litigation success | 6, 11, 1 | 5, 8, 12 |
| Spiritual | 9, 12 (6, 11) | |
- Krishnamurti's primary/supporting table by cusp (2nd..12th) exists in the Reader (e.g. 7th: Marriage 2, 11; Partnership 5, 11; Theft 2, 12; Danger from opponents 8, 12; 5th: Child birth 2, 11, Love marriage 7, 11, Speculation 2, 11 (10); 6th: Loan 2, 11, litigation 1, 11; 9th: long journey 3, 12, higher education 11 (4)). Will encode a machine-readable table with sources noted as "textbook convention, schools differ".
- Golden rule (KP): event needs promise + fulfilment houses and must exclude the denial houses (12th from the fulfilment houses).

### R5. Ruling planets (RP)
- Standard set: Day lord, Moon sign lord, Moon star lord, Lagna sign lord, Lagna star lord (5). Extended: + Moon sub lord and Lagna sub lord (7). Lagna sub lord is strongest, day lord weakest.
- Day lord is from sunrise to next sunrise (Hindu vara), not midnight.
- RP in the star/sub of another RP is powerful; RP in its own sub is very powerful; planet appearing 3+ times = strongest.
- Rahu/Ketu as RP: take the planet they represent (star lord / sign lord / conjunction). Use: judge at the moment of the question, confirm event timing, rectify birth time, choose muhurta.
- RPs change: ascendant ~1 deg per 4 min, so Lagna sub lord can flip within minutes.

### R6. Horary (Prashna) 1-249
- Querent gives a number 1-249 -> the numbered sub fixes the Lagna (sign, star, sub); the other cusps follow (Placidus) at the question moment/place; planets are those of the question moment.
- Which degree inside the sub: textbook sources vary (start of sub; or actual Lagna if it falls in that same sub). Option to expose: `lagna_point = start | middle | actual_if_inside`.
- Judgment: CSL of the house of the matter; RPs verify; dasha/bhukti of the chart at question time for timing; Moon sub lord checks "genuineness".

### R7. Ayanamsa (see machine probe above)
- KP old: Krishnamurti's table (flat per year, to the minute); KP New (KPNA) = continuous Newcomb-based, ~Lahiri - 4m54s (community-recommended; "avoid mixing ayanamsas"). Sources disagree on exact numbers (SEO sites contradict each other); we rely on swisseph modes 5 and 45 and say so.

### R8. Dasha timing (KP)
- Same Vimshottari, read through KP significators: Dasha -> Bhukti -> Antara (pratyantar) -> Sookshma (5th level Prana rarely). KP software prints "Dasa Days: 365.25" -> **year = 365.25 days**.
- Event happens in the **joint period** of dasha/bhukti/antara lords that signify the event houses (promise from CSL; fruitful = significators also among the RPs).
- Dasha lord's own **sub lord** gives permission: if it signifies the event houses (or at least not the denial houses) the period is allowed. Rule 9 (Four-step): a planet totally negating the event houses in step 3/4 will not fructify.
- Fruitful-significator selection methods: (1) detrimental-house method (drop planets that signify denial houses), (2) sub-lord method (planet whose sub lord signifies the event houses), (3) ruling-planet method (keep significators that are RPs).
- Dasha balance at birth from the Moon's exact longitude (nakshatra fraction remaining x lord's years); sub-period lengths = parent x child / 120.

### R9. Transit rules (KP, Reader 5)
- Transit only triggers what the running dasha/bhukti/antara already allow (trigger inside a period with right significators).
- Slow planets (Jupiter, Saturn, nodes) pick months; **Sun picks the week, Moon picks the day**: event falls when Sun / Moon (also Venus etc.) transit the sign, star or sub of the dasha/bhukti/antara/sookshma lords or of the RPs.
- Transit uses only two steps (sign lord, star lord); the transit planet's sub / sub's star is usually not used by Reader 5 (some use sub).
- Typical dwell times: Saturn ~13 months in a star; Rahu/Ketu ~8 months; Jupiter ~5 months 10 days; Sun 1-3 days per sub; many weeks for Saturn in a sub.
- Transit planet by matter: Moon dasha lord -> check Jupiter & Venus; job -> Sun & Saturn; property sale/purchase -> Mars.
- Retrograde: if dasha lord Saturn is itself retrograde in transit it can give results; a planet in the star of a retrograde planet cannot give results in its period (rule quoted in Four-step material; schools differ).
- Eclipse/combust: KP does not treat combustion as automatically crippling: a combust planet's significations still come from its star + sub lord.

### R10. Planet signification table (planet -> houses), the 'planet significator' view
- For each planet: houses signified through (1) its star lord's occupied house(s), (2) its star lord's owned house(s), (3) its own occupied house, (4) its own owned house(s) [order of strength 1..4 as per Four-step / Level 1-4 tables]. Rahu/Ketu add the houses of planets they conjoin/aspect, sign lord, star lord.
- Planet strength = its own star lord's strength; a planet in its own star is self-strength.
- 6-fold (Reader 6): planet in star of A and sub of C; adds sub-level houses; used to explain timing when 4-fold table misses a lord (e.g. Moon antara). Implement as option `include_sub_level`.
- Four-step theory (Shri Bhatt): step 1 planet's own houses, 2 star lord's houses, 3 sub lord's houses, 4 star lord of the sub lord; the same for the CSL; used for 'promise' and timing.

### R11. Nodes
- KP: Rahu/Ketu give results of conjoined planets first, then aspecting planets (some order: conjunction, aspect, star lord, sign lord ascending strength). The k_p_system forum endorses **Mean Rahu**; many software offer True. Expose `node_type = mean | true`, default mean, and tell the caller that sub lord of nodes can flip.
- Rahu/Ketu act as RPs by the planet they represent; a node is considered stronger than the planet it represents.

### R12. Birth-time rectification (KP)
- Rule: the RPs at the time of judgment (question about the event) must be among the RPs at birth/the event; compare the chart's Lagna sign/star/sub lord with the RP Moon sign/star/sub lord at all 3 levels; adjust birth time so the Lagna sub lord (and cusp sub lords of event cusps) match.
- Event-based: find times when the cusp sub lord of the event cusp links to the dasha lords at the event date; scan birth time +/- window in steps and score. This is the part we can compute (`rectify_birth_time` tool): scan Lagna sub boundaries, report candidate windows consistent with given events.

### R13. KP books (coverage checklist)
- Reader 1 Casting the Horoscope; 2 Fundamental Principles (incl. Fortuna, Rahu/Ketu, Uranus/Neptune); 3 Predictive Stellar Astrology; 4 Marriage, Married Life & Children; 5 Transits (Gocharaphala Nirnayam); 6 Horary Astrology (+ Medical Astrology, Profession, Twin birth, etc.).
- Fortuna (Part of Fortune) in KP = **Asc + Moon - Sun** (Reader 1 p.98); considered under research; include as an info point with its star/sub.
- KP uses 12 bodies in the textbook: 9 grahas + Uranus, Neptune, Pluto (optional; planned `include_outer = false` option).

### R14. Open research items (next rounds)
- KP dasha rules (dasha/bhukti/antara/sookshma significator chain; how "retrograde" and "combust" are treated; paridhi; sookshma timing).
- Transit rules in KP (only 2 steps: star lord and sign-lord; Sun/Moon/Jupiter/Saturn over sub of significators; Moon transit for day timing; Sun for month).
- Retrograde planets: result timing (delivers after turning direct / in transit when direct) and 'cuspal sub lord retrograde' rule.
- Rahu/Ketu details (mean vs true node in KP software), Lagna/12th/8th cuspal rules, Moon in KP (Moon as significator, Moon-dasha).
- Rectification via RPs and via sub-lord of event; Fortuna (Part of Fortune) in KP; Bhava-chalit/ cusp-vs-planet house placement; Nakshatra Nadi/Vimshottari balance.
- KP medical, financial, stock-market, electional (muhurta), mundane uses; Uranus/Neptune/Pluto (KP textbook mentions 12 bodies).
- Numerical facts to test: 249 table starts/ends; known KP chart cross-check (public example charts with listed cusps/subs).
