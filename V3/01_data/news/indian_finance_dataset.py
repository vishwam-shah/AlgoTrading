"""
indian_finance_dataset.py
=========================
Labeled training sentences for fine-tuning FinBERT on Indian stock market text.

Labels:
  0 = negative  (bearish, sell signal, bad news)
  1 = neutral   (informational, mixed, no clear direction)
  2 = positive  (bullish, buy signal, good news)

Target: 600+ samples, balanced across 3 classes.
Sectors: Banking, IT, Pharma, FMCG, Auto, Metals, Telecom, Power, Realty, Oil & Gas
"""

TRAINING_DATA = [

    # ══════════════════════════════════════════════════════════════════════════
    #  POSITIVE (label=2)
    # ══════════════════════════════════════════════════════════════════════════

    # Earnings beats — Banking
    ("HDFC Bank Q4 profit jumps 23 percent year on year beating estimates", 2),
    ("ICICI Bank reports record net interest income of 19000 crore", 2),
    ("SBI net profit rises 84 percent on lower provisions and strong credit", 2),
    ("Kotak Mahindra Bank beats street estimates asset quality improves sharply", 2),
    ("Axis Bank Q3 results impress with 62 percent rise in net profit", 2),
    ("Bank net interest margin expands 20 basis points above analyst forecast", 2),
    ("CASA ratio improves to 47 percent indicating low cost deposit strength", 2),
    ("Gross NPA falls to 1.2 percent lowest in eight quarters", 2),
    ("Provision coverage ratio improves to 82 percent signaling balance sheet strength", 2),
    ("Net NPA declines to 0.3 percent best in decade for private sector banks", 2),
    ("Retail loan book grows 22 percent driven by home auto and personal loans", 2),
    ("Credit growth accelerates to 16 percent year on year above industry average", 2),
    ("Slippage ratio falls to 0.8 percent indicating improving asset quality", 2),
    ("HDFC Bank advances grow 10 percent deposits surge 14 percent year on year", 2),
    ("Bank reports 5 percent sequential improvement in operating profit", 2),

    # Earnings beats — IT
    ("TCS Q4 revenue beats estimates deal wins at record 11 billion dollars", 2),
    ("Infosys raises FY26 guidance to 5 to 7 percent constant currency growth", 2),
    ("Wipro wins 1.5 billion dollar multi year outsourcing deal from US bank", 2),
    ("HCL Tech Q4 EBIT margin expands 150 basis points sequential improvement", 2),
    ("Tech Mahindra deal total contract value at 900 million beats expectations", 2),
    ("Infosys large deal wins at 2.5 billion dollars highest in 6 quarters", 2),
    ("Attrition falls to 11.8 percent lowest since pandemic for TCS", 2),
    ("IT sector headcount addition beats estimates demand recovery visible", 2),
    ("Persistent Systems revenue growth of 22 percent outperforms large peers", 2),
    ("Mphasis deal pipeline strongest in 5 years analyst upgrades to buy", 2),

    # Earnings beats — Pharma
    ("Sun Pharma US specialty revenue beats estimates chronic therapy growing", 2),
    ("Dr Reddys EBITDA margin improves 300 basis points on branded generics", 2),
    ("Cipla FY26 guidance raised on strong domestic formulations growth", 2),
    ("Divi's Laboratories contract manufacturing ramp up drives 18 percent revenue growth", 2),
    ("Aurobindo Pharma US approvals accelerate 40 ANDA clearances in quarter", 2),
    ("Lupin injectable portfolio gaining US market share margins improving", 2),
    ("Pharma company receives USFDA approval for complex generic drug", 2),
    ("Abbott India domestic business grows 14 percent outperforming market", 2),

    # Earnings beats — FMCG
    ("HUL volume growth of 6 percent beats expectations in rural market recovery", 2),
    ("Nestle India revenue grows 15 percent product premiumization working", 2),
    ("Britannia Q4 margin surprise on easing input costs wheat and sugar prices", 2),
    ("Dabur India rural recovery drives 9 percent volume growth beats estimates", 2),
    ("Marico operating margin improves 200 basis points on benign commodity prices", 2),
    ("ITC cigarette volumes ahead of expectations market share gains continue", 2),

    # Earnings beats — Auto
    ("Maruti Suzuki market share rises to 43 percent highest in four years", 2),
    ("Tata Motors Jaguar Land Rover EBIT margin beats estimates at 11.2 percent", 2),
    ("Hero MotoCorp rural demand recovery drives 8 percent volume growth", 2),
    ("Bajaj Auto exports surge 24 percent premium motorcycle segment strong", 2),
    ("Mahindra and Mahindra SUV waiting period indicates strong demand pipeline", 2),
    ("Two wheeler sales hit record high in festive season beats estimates", 2),

    # RBI / macro positive
    ("RBI cuts repo rate by 25 basis points boosting rate sensitive banking stocks", 2),
    ("RBI MPC adopts accommodative stance signals further rate cuts ahead", 2),
    ("India GDP grows at 7.2 percent beats IMF projection of 6.8 percent", 2),
    ("India PMI manufacturing at 57.5 highest in 16 months signals expansion", 2),
    ("GST collections hit record 1.87 lakh crore third consecutive month above 1.7 lakh crore", 2),
    ("India retail inflation falls to 4.2 percent within RBI comfort band", 2),
    ("Current account deficit narrows to 1.1 percent of GDP better than expected", 2),
    ("India forex reserves hit all time high at 670 billion dollars", 2),
    ("FDI inflows surge 35 percent as India emerges as preferred investment destination", 2),
    ("Moody's upgrades India growth forecast to 7 percent for FY27", 2),
    ("RBI allows higher FPI limits in corporate bonds positive for debt markets", 2),
    ("India IIP industrial production rises 5.5 percent beats estimate of 4.2", 2),
    ("Trade deficit narrows sharply on record services exports", 2),

    # FII / institutional positive
    ("FIIs net buyers of 8000 crore in Indian equities fifth consecutive session", 2),
    ("Domestic mutual funds add 17000 crore to banking stocks buy on dip", 2),
    ("FPI inflows into India equity touch 50000 crore in single month", 2),
    ("MSCI India weight increase to trigger passive FII inflows of 3 billion", 2),
    ("LIC increases stake in HDFC Bank to 9 percent positive signal", 2),
    ("Sovereign wealth fund buys 1.5 percent stake in Indian bank at premium", 2),
    ("Block deal of 3000 crore placed successfully demand from long only funds", 2),
    ("Domestic institutional investors net buyers for 15 consecutive sessions", 2),

    # Technical / price positive
    ("Stock breaks out above resistance on 3x volume bullish pattern", 2),
    ("Golden cross formed 50 day EMA crosses above 200 day EMA", 2),
    ("Nifty 50 clears 23000 with strong momentum indicators", 2),
    ("Sensex hits all time high of 80000 led by banking and IT", 2),
    ("Nifty Bank index surges 2.5 percent outperforms broader Nifty 50", 2),
    ("Stock hits 52 week high on strong volumes breakout confirmed", 2),
    ("RSI at 65 with positive divergence bullish momentum building", 2),
    ("Market breadth strong 1900 advances vs 500 declines on NSE", 2),

    # F&O positive signals
    ("Put call ratio falls to 0.65 strong bullish positioning in options market", 2),
    ("Nifty options shorts covering at 22000 put writers buying back", 2),
    ("Open interest buildup in calls above market level bullish signal", 2),
    ("Nifty Bank call writers unwinding at 47000 strike short squeeze likely", 2),
    ("Implied volatility collapses post results stock set for re-rating", 2),
    ("Futures basis turns positive premium expands bullish rollover", 2),
    ("Nifty roll cost jumps indicating longs carrying positions to next series", 2),

    # Corporate actions positive
    ("Board approves 10000 crore share buyback at 15 percent premium to CMP", 2),
    ("Company declares dividend of 22 rupees per share for FY26", 2),
    ("Promoter increases stake by 3 percent in open market purchases", 2),
    ("Stock added to MSCI EM index passive buying expected", 2),
    ("Credit rating upgraded from AA to AAA by CRISIL India", 2),
    ("Goldman Sachs initiates with buy rating and 2200 rupee price target", 2),
    ("Morgan Stanley raises target price to 1950 after strong quarterly results", 2),
    ("Macquarie upgrades to outperform raises earnings estimates by 12 percent", 2),
    ("Company announces demerger creating unlocked value for shareholders", 2),
    ("Strategic investor acquires 10 percent stake at 20 percent premium to CMP", 2),

    # Metals / Commodities positive
    ("Tata Steel Europe operations return to profit on higher steel realizations", 2),
    ("JSW Steel crude steel production hits record high demand from infra sector", 2),
    ("Hindalco aluminium margins expand on LME price rally and cost efficiencies", 2),
    ("NMDC iron ore volumes hit record quarterly high on strong demand", 2),
    ("Coal India production exceeds target beats analyst volume estimates", 2),

    # Power / Infra positive
    ("Adani Green commissions 1000 MW solar capacity ahead of schedule", 2),
    ("Power Grid transmission revenue grows 12 percent on new asset additions", 2),
    ("NTPC PAF plant availability factor improves to record 94 percent", 2),
    ("L&T order inflows surge 25 percent largest quarterly intake in 5 years", 2),
    ("RVNL secures 5000 crore railway project order stock rallies 4 percent", 2),

    # Telecom positive
    ("Reliance Jio ARPU rises to 210 rupees on tariff hike taking effect", 2),
    ("Bharti Airtel India wireless revenue beats estimates ARPU at 245 rupees", 2),
    ("Telecom sector ARPU recovery trend intact industry structure improving", 2),

    # Oil & Gas positive
    ("ONGC oil discovery of significant reserve in Krishna Godavari basin", 2),
    ("Reliance Industries O2C margins improve sharply on product spread expansion", 2),
    ("Petronet LNG volume growth 18 percent on new regasification contracts", 2),

    # Realty positive
    ("DLF pre-sales hit record 15000 crore luxury housing demand strong", 2),
    ("Godrej Properties launches 3000 crore residential project sold out in 2 days", 2),
    ("Housing demand in top 8 cities grows 19 percent year on year record launches", 2),

    # ══════════════════════════════════════════════════════════════════════════
    #  NEGATIVE (label=0)
    # ══════════════════════════════════════════════════════════════════════════

    # Earnings misses — Banking
    ("HDFC Bank Q4 profit misses estimates net interest margin compresses", 0),
    ("SBI gross NPA surges to 5.1 percent from 3.8 percent previous quarter", 0),
    ("ICICI Bank provisions spike 45 percent on surprise corporate slippages", 0),
    ("Yes Bank reports loss as RBI mandated provisions wipe out income", 0),
    ("Axis Bank watchlist accounts double raising concern of future NPA pain", 0),
    ("Bank net interest margin falls 15 basis points below street estimates", 0),
    ("CASA ratio deteriorates as customers shift deposits to higher yield FDs", 0),
    ("Credit to deposit ratio rises to 82 percent raising liquidity concerns", 0),
    ("Kotak Bank RBI restricts new customer onboarding via online channels", 0),
    ("Bank chairman resigns amid governance concerns board reconstitution due", 0),
    ("RBI orders forensic audit into bank lending practices major risk", 0),
    ("Microfinance loan stress spreads to mid size private sector banks", 0),

    # Earnings misses — IT
    ("Infosys cuts FY26 revenue guidance to 1 to 3 percent from 4 to 7 percent", 0),
    ("TCS headcount falls 5000 in quarter on weak demand environment", 0),
    ("Wipro Q3 revenue misses guidance company issues weak Q4 outlook", 0),
    ("Attrition rises to 23.8 percent highest level since post pandemic boom", 0),
    ("HCL Tech EBIT margin contracts 280 basis points on wage hike impact", 0),
    ("IT sector deal wins miss estimates at 2 billion vs expected 3.5 billion", 0),
    ("US client discretionary IT spend paused amid recession concerns", 0),
    ("Tech Mahindra quarterly loss on legacy restructuring charges", 0),
    ("IT company loses key client worth 200 million dollar annual revenue", 0),

    # Earnings misses — Pharma
    ("Sun Pharma US revenue disappoints on generic price erosion", 0),
    ("USFDA issues import alert for company's key manufacturing plant", 0),
    ("Cipla receives complete response letter from USFDA for injectable product", 0),
    ("Divi's Laboratories EBITDA margin compresses on higher input costs", 0),
    ("Lupin US base business erosion continues 15 percent quarterly decline", 0),
    ("USFDA warning letter issued to pharma plant major regulatory risk", 0),
    ("Drug company faces patent cliff as blockbuster product loses exclusivity", 0),

    # Earnings misses — FMCG
    ("HUL volume growth disappoints at 2 percent below estimates of 5 percent", 0),
    ("Nestle India margin pressure from higher cocoa prices misses estimates", 0),
    ("Marico Parachute volumes decline 3 percent on rural demand slowdown", 0),
    ("FMCG companies face margin compression as input costs remain elevated", 0),

    # Earnings misses — Auto
    ("Maruti Suzuki Q4 profit falls 15 percent on discounting and cost pressure", 0),
    ("Tata Motors JLR supply chain disruption hits volumes guidance lowered", 0),
    ("Hero MotoCorp volumes disappoint rural demand recovery stalling", 0),
    ("Auto sector wholesale volumes miss estimates retail channel destocking", 0),
    ("EV sales slowdown weighs on auto stocks demand below expectation", 0),

    # RBI / macro negative
    ("RBI hikes repo rate by 50 basis points to combat sticky inflation", 0),
    ("India inflation spikes to 7.4 percent breaching RBI upper band sharply", 0),
    ("RBI restricts bank business citing serious IT governance failures", 0),
    ("SEBI bans promoter from capital markets for 3 years insider trading", 0),
    ("India GDP slows to 5.6 percent below consensus estimate of 6.5 percent", 0),
    ("Current account deficit widens to 3.2 percent of GDP trade balance worsens", 0),
    ("Rupee hits all time low of 88 versus dollar capital outflows accelerate", 0),
    ("FII outflows cross 30000 crore in January as dollar strengthens globally", 0),
    ("India fiscal deficit widens beyond target government spending overshoots", 0),
    ("RBI flags systemic risk in microfinance sector stress spreading", 0),
    ("India PMI services falls to 49.8 contraction territory disappoints", 0),

    # FII / institutional negative
    ("FIIs dump 15000 crore in single session largest outflow in 6 months", 0),
    ("MSCI cuts India weight triggering estimated 2 billion passive FII selling", 0),
    ("Emerging market ETF outflows hit 5 billion dollars India impacted", 0),
    ("Foreign investors exit India on rising dollar and US rate concerns", 0),
    ("FPI sell off in banking stocks on NPA concern contagion fear", 0),

    # Technical / price negative
    ("Nifty breaks below 200 day EMA on high volume death cross forming", 0),
    ("Sensex crashes 1500 points worst single day fall in 18 months", 0),
    ("Market breadth deeply negative 2400 declines versus 300 advances", 0),
    ("Stock hits 52 week low as institutional selling accelerates", 0),
    ("Nifty Bank index breaks key 44000 support downside open", 0),
    ("Circuit breaker triggered stock crashes 20 percent on NSE", 0),
    ("Bearish engulfing candle on weekly chart major reversal signal", 0),

    # F&O negative signals
    ("Put call ratio spikes to 1.9 extreme bearish positioning in Nifty", 0),
    ("India VIX surges 45 percent in single session extreme fear indicator", 0),
    ("Nifty put writers covering at 21500 support level breaks down", 0),
    ("High open interest buildup in puts signals institutional hedging activity", 0),
    ("Futures discount widens bearish rollover longs exiting positions", 0),
    ("VIX above 28 fear elevated caution advised reduced exposure recommended", 0),

    # Corporate negative
    ("Company defaults on 500 crore NCD repayment solvency risk emerges", 0),
    ("Promoter pledges 60 percent shareholding debt overhang major concern", 0),
    ("ICRA downgrades company to BBB minus from A credit quality deteriorating", 0),
    ("SEBI issues show cause notice for insider trading violations", 0),
    ("Income tax department conducts searches at company headquarters Bangalore", 0),
    ("Company withdraws IPO citing adverse market conditions", 0),
    ("Company announces rights issue at 40 percent discount to CMP dilution", 0),
    ("CEO quits abruptly amid board dispute governance red flag", 0),
    ("Stock market crash HDFC Bank LIC ICICI Bank among 122 hitting 52 week low", 0),
    ("Forensic audit reveals revenue recognition irregularities management change", 0),
    ("Company loses arbitration order 2000 crore payable to government", 0),

    # Metals / Commodities negative
    ("Tata Steel Europe reports loss as energy costs and weak demand bite", 0),
    ("Steel prices fall 8 percent as China exports flood global markets", 0),
    ("Hindalco aluminium margin compresses as LME price falls to 18 month low", 0),
    ("Coal India volumes miss target on logistical constraints and rain disruption", 0),

    # Power / Infra negative
    ("Power sector stranded assets problem resurfaces as discoms default on dues", 0),
    ("L&T order inflows miss estimates as government capex slows", 0),
    ("Adani Green faces USFDA scrutiny equivalent charges bribery allegations", 0),
    ("NTPC forced outage at major plant impacts generation capacity", 0),

    # Telecom negative
    ("Vodafone Idea AGR dues uncertainty threatens survival of company", 0),
    ("Telecom sector faces spectrum auction outgo pressure on balance sheets", 0),
    ("Jio 5G capex requirement higher than expected margins under pressure", 0),

    # Oil & Gas negative
    ("OMCs marketing margins under pressure as crude prices spike above 100 dollars", 0),
    ("ONGC production disappoints aging field natural decline rate worsens", 0),
    ("Reliance Industries refining margins compress sharply on product oversupply", 0),

    # Realty negative
    ("Real estate launches slow sharply as home loan rates rise above 9 percent", 0),
    ("DLF unsold inventory rises as premium housing demand slows", 0),
    ("NBFC real estate lending tightens as RBI flags sector risks", 0),

    # Global negative
    ("US Fed signals prolonged higher rates risk off sentiment hits India", 0),
    ("China property sector crisis triggers global risk aversion FII selling", 0),
    ("Brent crude spikes above 100 dollars inflation fears return India impacted", 0),
    ("US recession risk rises PMI contracts India IT sector exposed", 0),
    ("Global banking crisis spreads to Asian markets India VIX spikes", 0),

    # ══════════════════════════════════════════════════════════════════════════
    #  NEUTRAL (label=1)
    # ══════════════════════════════════════════════════════════════════════════

    # Informational / scheduled events
    ("HDFC Bank board meeting scheduled April 18 to consider Q4 FY26 results", 1),
    ("TCS to announce quarterly results on April 10 2026", 1),
    ("RBI monetary policy committee meeting scheduled for June 5 to 7", 1),
    ("NSE announces F&O lot size revision effective next expiry series", 1),
    ("BSE Sensex changes composition 2 stocks added 2 removed", 1),
    ("Stock market holiday on April 14 Ambedkar Jayanti NSE BSE closed", 1),
    ("SEBI extends comment deadline on new regulations by 30 days", 1),
    ("Nifty 50 options expiry today elevated volatility expected at open", 1),
    ("Company annual general meeting scheduled for July 15 in Mumbai", 1),
    ("RBI publishes monthly bulletin on credit and monetary data", 1),
    ("NSE derivative monthly expiry rollover data shows mixed open interest", 1),
    ("Futures expiry rollover in progress open interest shifts to next series", 1),
    ("FII data not available US markets closed for Thanksgiving holiday", 1),
    ("Company files quarterly shareholding pattern with BSE and NSE", 1),
    ("Index rebalancing impact to be absorbed over next two trading sessions", 1),
    ("Nifty 50 reconstitution effective from June 28 two stocks replaced", 1),
    ("Provisional trade data for March released imports and exports mixed", 1),

    # Price and volume tracking
    ("HDFC Bank share price today live NSE BSE stock price chart update", 1),
    ("HDFC Bank trades flat in early trade investors await Q4 results", 1),
    ("Stocks to watch HDFC Bank Reliance TCS Infosys tomorrow opening", 1),
    ("Mid session banking stocks consolidating ahead of RBI policy decision", 1),
    ("Nifty 50 moves in narrow 100 point range global cues mixed", 1),
    ("Sensex opens flat as traders await US Fed minutes later today", 1),
    ("HDFC Bank among top stocks mutual funds bought and sold in March 2026", 1),
    ("Nifty Bank relatively flat underperforms broader market marginally", 1),
    ("Indian market opens cautiously global cues mixed Asia divergent", 1),
    ("Gift Nifty suggests flat opening domestic data awaited", 1),
    ("Market in wait and watch mode ahead of quarterly earnings season", 1),
    ("Volumes below 10 day average no clear directional bias yet", 1),
    ("Options market shows straddle strategy pricing 1 percent move either way", 1),

    # Mixed results
    ("Strong revenue growth but margin contraction disappoints analysts mixed reaction", 1),
    ("Bank reports beat on NII but miss on fee income overall results mixed", 1),
    ("IT company revenue in line guidance maintained cautious on demand outlook", 1),
    ("Pharma results good domestic strong but US generics continue to erode", 1),
    ("FII sellers offset by strong DII buying Nifty holds key 22000 support", 1),
    ("Profit booking at resistance despite good results stock consolidates", 1),
    ("Good asset quality but net interest margin under pressure near term", 1),
    ("Deal wins strong but attrition still elevated FY26 guidance in line", 1),
    ("Revenue beats but operating leverage yet to play out margins flat", 1),
    ("Management positive on long term but near term headwinds acknowledged", 1),
    ("Auto volume ahead but realizations disappoint average selling price flat", 1),
    ("FMCG volume growth recovers but pricing power not yet visible", 1),

    # Analyst views / ratings
    ("Analyst maintains neutral rating on HDFC Bank with target of 1750", 1),
    ("Goldman Sachs keeps market perform on IT sector sector in line view", 1),
    ("Brokerages split on banking sector bulls and bears equally divided", 1),
    ("Analyst cuts target by 5 percent maintains hold recommendation", 1),
    ("CLSA retains reduce rating but acknowledges improving fundamentals", 1),
    ("Street consensus target price for Nifty 50 at 24000 by December", 1),
    ("Rating agency maintains stable outlook on banking sector no change", 1),

    # Corporate announcements (routine)
    ("Company announces board meeting to consider interim dividend", 1),
    ("Promoter sells 1 percent stake in open market to meet personal needs", 1),
    ("Company completes share buyback buying back 80 percent of target quantity", 1),
    ("Company announces rights issue to raise 2000 crore at 5 percent discount", 1),
    ("Subsidiary listing process initiated regulatory filings underway", 1),
    ("Company enters into MOU with state government for new plant", 1),
    ("Board approves raising 5000 crore through NCDs for general purposes", 1),
    ("HDFC Bank appoints law firm to review governance and compliance framework", 1),
    ("Company changes name following completed merger integration", 1),

    # Global cues (mixed)
    ("US markets end mixed S&P flat Nasdaq up 0.3 percent Dow down 0.2 percent", 1),
    ("Asian markets diverge Japan up China down India to take mixed cues", 1),
    ("Brent crude steady at 84 dollars no major move energy stocks await direction", 1),
    ("Dollar index DXY flat no major currency moves overnight EM stable", 1),
    ("US Fed minutes show divided committee no clear rate path signal", 1),
    ("IMF retains India growth forecast unchanged at 6.8 percent for FY26", 1),
    ("Nikkei gains 0.5 percent Hang Seng falls 0.4 percent mixed Asia cues", 1),
    ("VIX at 18 below 20 normal range no immediate fear or greed signal", 1),

    # Macro data (neutral)
    ("India CPI inflation for March to be released on April 12 estimates 4.8", 1),
    ("India trade data shows exports flat imports slightly higher deficit stable", 1),
    ("RBI keeps repo rate unchanged at 6.5 percent as expected by 95 percent of analysts", 1),
    ("WPI wholesale price index for February comes in at 0.2 percent in line", 1),
    ("India Q3 GDP to be released Friday consensus estimate at 6.7 percent", 1),

    # Sector rotation (neutral)
    ("Investors rotate from IT to banking stocks as rate cycle turns", 1),
    ("Midcap outperforms largecap today sector rotation to value themes", 1),
    ("PSU banks gain while private banks give up intraday sector divergence", 1),
    ("Defensive FMCG sectors bid up on global uncertainty risk off mode", 1),
    ("HDFC Bank trades in green as broader banking index mixed performance", 1),

    # Short phrases and headlines (common in news feeds)
    ("Vodafone Idea Ola Electric YES Bank HDFC Bank stocks to watch today", 1),
    ("HDFC Bank share price NSE BSE live updates today", 1),
    ("Top 10 stocks to buy today as per technical analysis HDFC Bank included", 1),
    ("Sensex Nifty outlook today Gift Nifty cues and key levels to watch", 1),
    ("Stocks in news HDFC Bank SBI ICICI Bank TCS Infosys", 1),
    ("Markets today Nifty Bank outlook key support resistance levels", 1),
    ("Q4 results season begins banking IT pharma earnings calendar", 1),


    # ══════════════════════════════════════════════════════════════════════════
    #  BATCH 2 — 300 additional examples targeting neutral gap + new sectors
    # ══════════════════════════════════════════════════════════════════════════

    # ── NEUTRAL — pre-result / scheduled events (most confused class) ─────────
    ("Infosys Q4 results tomorrow street estimates at 4.5 percent revenue growth", 1),
    ("HDFC Bank earnings call scheduled for 6 PM today analysts on watch", 1),
    ("Wipro board meeting on Friday to consider quarterly dividend declaration", 1),
    ("SBI Q3 results due next week consensus EPS estimate at 19 rupees", 1),
    ("Reliance Industries AGM on July 18 product launches expected", 1),
    ("Tata Motors to report Q2 results after market hours today", 1),
    ("Sun Pharma results due Thursday USFDA update also awaited", 1),
    ("Bajaj Finance Q4 scheduled April 29 AUM growth in focus", 1),
    ("Kotak Mahindra Bank board meets next week to consider fundraising", 1),
    ("Maruti Suzuki sales data for March to be released on April 1", 1),

    # ── NEUTRAL — consolidation and range-bound trading ───────────────────────
    ("Stock consolidates near 52 week high awaiting fresh trigger", 1),
    ("Nifty Bank trades sideways for third session no directional breakout", 1),
    ("HDFC Bank moves in narrow 10 rupee range ahead of earnings", 1),
    ("Market consolidates after 8 percent run up in one month pause expected", 1),
    ("IT stocks sideways as deal pipeline awaited no fresh catalyst", 1),
    ("Stock at key support of 200 day EMA no clear breakout or breakdown", 1),
    ("Sensex ranges between 79500 and 80200 traders prefer sidelines", 1),
    ("Stock holds at prior breakout level retest in progress direction unclear", 1),
    ("Banking index trades in 300 point range no institutional conviction", 1),
    ("Large cap stocks flat while midcaps diverge broader trend unclear", 1),

    # ── NEUTRAL — hold ratings and neutral analyst views ─────────────────────
    ("Citi maintains neutral on HDFC Bank target price unchanged at 1750", 1),
    ("Morgan Stanley keeps equal weight on Infosys no change to estimates", 1),
    ("CLSA retains hold on Wipro near term demand outlook unclear", 1),
    ("Emkay maintains accumulate on SBI at current levels target 850", 1),
    ("Kotak Securities hold on Bharti Airtel awaiting tariff hike impact", 1),
    ("UBS neutral on TCS valuation full near term upside limited", 1),
    ("Prabhudas Lilladher hold rating on Maruti target 12500 unchanged", 1),
    ("Analyst maintains market perform on Bajaj Finance premium valuation", 1),
    ("ICICI Securities hold on ITC tax policy uncertainty overhang", 1),
    ("Jefferies neutral on Tata Motors JLR margin recovery priced in", 1),

    # ── NEUTRAL — macro data in-line ─────────────────────────────────────────
    ("India CPI inflation for April at 4.85 percent in line with estimates", 1),
    ("India IIP for February at 4.9 percent broadly in line consensus 5 percent", 1),
    ("WPI inflation for March at 0.53 percent no major surprise", 1),
    ("India trade deficit for March at 15.6 billion dollars near estimates", 1),
    ("RBI keeps policy rates unchanged in line with unanimous street expectation", 1),
    ("India Q2 GDP at 6.7 percent in line with consensus estimate of 6.8 percent", 1),
    ("India PMI services at 58.5 strong but slightly below last month 59.1", 1),
    ("Core sector output growth of 4.2 percent broadly in line with forecasts", 1),
    ("India direct tax collections on track to meet full year target", 1),
    ("India fiscal deficit at 52 percent of target at half year mark as expected", 1),

    # ── NEUTRAL — FII DII data (no strong direction) ──────────────────────────
    ("FII net sellers of 200 crore DII buy 500 crore net market flat", 1),
    ("Provisional FII data shows net purchase of 50 crore negligible flow", 1),
    ("DII buying offsets FII selling for second consecutive session", 1),
    ("FII activity muted in holiday shortened week no clear trend", 1),
    ("Mutual fund SIP data shows steady 19000 crore monthly inflow no change", 1),
    ("FPI category wise data shows buying in financials selling in IT mixed", 1),
    ("Institutional activity light ahead of long weekend no major flows", 1),
    ("FII index futures net short position reduces marginally covers minor", 1),

    # ── NEUTRAL — F&O and derivatives data ───────────────────────────────────
    ("Nifty PCR at 1.1 neutral zone no extreme fear or greed signal", 1),
    ("Nifty open interest flat expiry rollover in progress no bias", 1),
    ("India VIX at 14.5 normal range options fairly priced both ways", 1),
    ("Nifty straddle at 22000 priced at 200 points 0.9 percent move expected", 1),
    ("Bank Nifty rollover at 72 percent near 3 month average no signal", 1),
    ("Monthly F&O expiry tomorrow elevated volumes expected no bias", 1),
    ("Options chain shows max open interest at 22000 call and 21500 put", 1),
    ("Nifty futures premium at 45 points cost of carry normal no signal", 1),
    ("HDFC Bank futures open interest unchanged no fresh build up", 1),
    ("Implied volatility flat after results event risk removed stock settles", 1),

    # ── NEUTRAL — global cues mixed ───────────────────────────────────────────
    ("Dow Jones ends flat S&P 500 gains 0.1 percent mixed close Wall Street", 1),
    ("Asian markets diverge Japan gains 0.8 percent China falls 0.5 percent", 1),
    ("US dollar index steady India rupee flat at 83.5 no major move", 1),
    ("Brent crude at 82 dollars holding range no catalyst either way", 1),
    ("US Fed holds rates unchanged as expected commentary balanced not hawkish", 1),
    ("China PMI manufacturing at 50.1 barely expansionary mixed signal", 1),
    ("ECB keeps rates unchanged euro flat global bonds muted", 1),
    ("Nikkei gains 0.3 percent SGX Nifty up 0.1 percent tepid Asia cues", 1),
    ("US CPI in line with expectations neither hot nor cold markets shrug", 1),
    ("IMF World Economic Outlook maintains India forecast unchanged at 6.8", 1),

    # ── NEUTRAL — routine corporate disclosures ───────────────────────────────
    ("Company files DRHP with SEBI for proposed IPO details awaited", 1),
    ("HDFC Bank changes record date for dividend entitlement to June 14", 1),
    ("Subsidiary board approves raising 1000 crore NCD allotment routine", 1),
    ("Company changes registered office address no operational impact", 1),
    ("Quarterly shareholding pattern shows no change in promoter holding", 1),
    ("Company completes allotment of shares under ESOP scheme as planned", 1),
    ("Credit rating agency affirms AA rating with stable outlook no change", 1),
    ("Company clarifies newspaper report saying it is routine business inquiry", 1),
    ("Stock exchange seeks clarification company says nothing material", 1),
    ("Insider trading window opens as results blackout period ends", 1),

    # ── NEUTRAL — sector rotation and relative performance ───────────────────
    ("Banking outperforms IT today sector rotation on rate cut hope", 1),
    ("Defensive FMCG gains as markets turn cautious risk off lite", 1),
    ("Midcap indices diverge from largecap no broad conclusion", 1),
    ("PSU banks outperform private banks intraday no fresh news", 1),
    ("Realty stocks move as interest rate sensitives rotate into sector", 1),
    ("Nifty IT underperforms Nifty 50 by 0.4 percent today mild drag", 1),
    ("Auto index flat despite good wholesale data investors await retail", 1),
    ("Metal stocks pause after 5 percent rally profit taking at resistance", 1),
    ("Healthcare index stable defensive buying offsets growth stock selling", 1),
    ("Smallcap index underperforms today broader market risk off", 1),

    # ── NEUTRAL — short ticker-style headlines ────────────────────────────────
    ("SBI share price NSE today", 1),
    ("TCS Q4 results live updates", 1),
    ("Reliance Industries stock price today BSE NSE", 1),
    ("Nifty 50 live chart and updates", 1),
    ("Sensex today opening levels", 1),
    ("ICICI Bank share price chart", 1),
    ("Top gainers and losers on NSE today", 1),
    ("Nifty Bank index live performance", 1),
    ("Infosys share price today NSE", 1),
    ("Bajaj Finance stock update", 1),
    ("HUL share price BSE NSE", 1),
    ("Adani Enterprises stock price live", 1),
    ("Kotak Bank results live update blog", 1),
    ("Weekly market outlook Nifty Sensex key levels", 1),
    ("Nifty midcap 150 performance update", 1),

    # ── POSITIVE — PSU banks ──────────────────────────────────────────────────
    ("State Bank of India reports best quarterly profit in decade asset quality best ever", 2),
    ("SBI credit growth at 15 percent NPA decline continues strong outlook", 2),
    ("Bank of Baroda Q4 ROE improves to 15 percent ahead of estimates", 2),
    ("Punjab National Bank net NPA falls to 0.7 percent massive turnaround", 2),
    ("Canara Bank provisions fall sharply as stressed asset resolution accelerates", 2),
    ("PSU banks re-rated as NPA cycle ends credit cost near historical low", 2),
    ("SBI Cards Q4 profit beats estimates spends growth strong at 18 percent", 2),
    ("Bank of India rights issue oversubscribed 4 times strong demand signal", 2),

    # ── POSITIVE — Consumer Durables / Electronics ────────────────────────────
    ("Dixon Technologies wins Apple assembly contract for India production", 2),
    ("Voltas room AC market share rises to 24 percent summer season strong", 2),
    ("Havells India Q4 beats estimates cable and wiring margin expands", 2),
    ("Blue Star commercial AC segment grows 32 percent beats street estimate", 2),
    ("Amber Enterprises RAC volume growth 28 percent PLI benefits visible", 2),
    ("Titan Company jewellery revenue grows 22 percent festive demand robust", 2),
    ("Vedant Fashions Manyavar ethnic wear demand recovery sharp after dip", 2),

    # ── POSITIVE — Specialty Chemicals ───────────────────────────────────────
    ("PI Industries agro-chemicals export order book at record high FY27 strong", 2),
    ("SRF fluorochemicals segment margin expands 400 basis points above estimates", 2),
    ("Navin Fluorine new CRAMS facility commissioned ahead of schedule", 2),
    ("Vinati Organics Q4 revenue beats antioxidant demand surge", 2),
    ("Aarti Industries order backlog hits 5000 crore strongest pipeline ever", 2),
    ("Deepak Nitrite phenol margins recover sharply pricing power returns", 2),

    # ── POSITIVE — Healthcare / Hospitals ─────────────────────────────────────
    ("Apollo Hospitals Q4 EBITDA per bed improves to record 1.4 lakh rupees", 2),
    ("Fortis Healthcare occupancy at 70 percent highest in 8 quarters", 2),
    ("Max Healthcare bed capacity expansion complete occupancy ramps up", 2),
    ("Narayana Hrudayalaya international business grows 40 percent Cayman strong", 2),
    ("Healthcare sector re-rating underway as post-Covid normalcy restored", 2),

    # ── POSITIVE — Capital Goods / Infra ─────────────────────────────────────
    ("Siemens India order inflows surge 40 percent energy transition capex strong", 2),
    ("ABB India Q4 beats estimates execution momentum strongest in 5 years", 2),
    ("Thermax order backlog at record 12000 crore industrial capex cycle turning", 2),
    ("Bharat Electronics wins 4500 crore defence order export pipeline building", 2),
    ("Hindustan Aeronautics order book 94000 crore 5 year revenue visibility", 2),
    ("Government capex push boosts L&T RVNL IRB Infra stocks rally", 2),
    ("IRB Infrastructure toll collections hit record high traffic growth strong", 2),

    # ── POSITIVE — agri / rural themes ───────────────────────────────────────
    ("Normal monsoon forecast by IMD positive for rural demand FMCG auto", 2),
    ("Rabi crop sowing ahead of last year rural income recovery on track", 2),
    ("Kharif sowing 8 percent above normal reservoir levels comfortable", 2),
    ("UPL agrochemicals export revival drives earnings beat stock rallies", 2),
    ("Rural consumption recovery visible in entry level two wheeler sales data", 2),

    # ── POSITIVE — broader positive market signals ────────────────────────────
    ("India added to JPMorgan EM bond index 25 billion dollar inflow expected", 2),
    ("Moody's upgrades India sovereign outlook to positive from stable", 2),
    ("Standard and Poors raises India growth forecast to 7.1 percent", 2),
    ("World Bank praises India infrastructure progress upgrades investment grade", 2),
    ("India manufacturing PMI at 59.1 highest in 15 years expansion strong", 2),
    ("India receives record FDI of 85 billion dollars in FY26", 2),
    ("Government announces 10 lakh crore infrastructure budget allocation", 2),
    ("Production linked incentive scheme drives manufacturing sector boom", 2),
    ("India overtakes UK as 5th largest economy milestone reached ahead of time", 2),
    ("SEBI simplifies IPO listing process boosts market confidence", 2),

    # ── NEGATIVE — PSU / governance issues ────────────────────────────────────
    ("PMC Bank depositors face hardship RBI extends withdrawal restrictions", 0),
    ("PSU bank under PCA prompt corrective action RBI imposes restrictions", 0),
    ("Government bank recapitalization bonds signal PSU bank balance sheet stress", 0),
    ("Punjab and Maharashtra Bank fraud 6500 crore depositors money at risk", 0),
    ("PSU bank reports divergence RBI finds NPA underreporting by 3000 crore", 0),

    # ── NEGATIVE — consumer stress ────────────────────────────────────────────
    ("Rural consumption slows sharply heat wave and erratic monsoon impact demand", 0),
    ("FMCG volume contraction in rural India third consecutive quarter of decline", 0),
    ("Consumer confidence index falls to lowest in two years belt tightening", 0),
    ("Credit card delinquency rates rise to 3 year high stress in retail loans", 0),
    ("Two wheeler and entry level car sales miss estimate rural distress visible", 0),
    ("Unsecured loan NPA rise rapidly across banks and NBFCs systemic concern", 0),
    ("Microfinance sector stress deepens borrower overlap and overleveraging issue", 0),

    # ── NEGATIVE — global spillover ───────────────────────────────────────────
    ("US recession fears trigger global selloff India Nifty falls 3 percent", 0),
    ("China devalues yuan competitive pressure on Indian exporters increases", 0),
    ("OPEC production cut pushes Brent to 95 dollars India import bill swells", 0),
    ("US Treasury yields spike to 5 percent EM capital outflows intensify", 0),
    ("Silicon Valley Bank collapse triggers global banking risk off sentiment", 0),
    ("Geopolitical tension in Middle East spikes crude oil safe haven buying", 0),
    ("Dollar index DXY surges to 107 rupee pressure FII selling accelerates", 0),
    ("Global trade slowdown hits IT sector deal wins miss across the board", 0),

    # ── NEGATIVE — regulatory / legal ────────────────────────────────────────
    ("SEBI bans top mutual fund manager for front running investor harm", 0),
    ("NSE co-location scam resurfaces new charges against exchange officials", 0),
    ("Company fined 500 crore by CCI for anti-competitive practices", 0),
    ("ED attachment order on company assets business disruption likely", 0),
    ("NCLT admits insolvency petition against prominent infrastructure company", 0),
    ("SEBI orders disgorgement of 200 crore from promoter insider trading", 0),
    ("Ministry of corporate affairs investigation launched at company premises", 0),

    # ── NEGATIVE — pharma / biotech ───────────────────────────────────────────
    ("USFDA issues form 483 with 7 observations serious regulatory concern", 0),
    ("Plant shutdown ordered by USFDA 30 percent of US revenue at risk", 0),
    ("Biocon biosimilars US launch delayed FDA needs additional data", 0),
    ("Generic drug price erosion accelerates 12 percent quarterly decline US", 0),
    ("API active pharmaceutical ingredient price spike hurts formulation margins", 0),
    ("Key molecule patent loses exclusivity revenue cliff ahead for pharma co", 0),

    # ── NEGATIVE — broader negative signals ──────────────────────────────────
    ("India rating watch negative Fitch flags fiscal slippage risk", 0),
    ("Foreign exchange reserves fall 15 billion in 3 weeks RBI intervening", 0),
    ("Bank credit growth slows to 9 percent lowest in 5 years demand weak", 0),
    ("Power sector discom losses mount state government finances strained", 0),
    ("Real estate launches fall 30 percent as home loan rates bite buyers", 0),
    ("Sugar mills face Rs 8000 crore cane arrears payment crisis deepens", 0),
    ("Steel imports from China surge 40 percent dumping concern for JSW Tata", 0),
    ("Airline sector cash burn accelerates ATF fuel cost spike threatens viability", 0),
    ("Textile export orders cancel as EU slowdown bites garment sector", 0),
    ("Startup funding winter deepens 60 percent drop in VC investments", 0),

    # ══════════════════════════════════════════════════════════════════════════
    #  BATCH 3 — 180 hard edge-case examples targeting 87.8% → 90%
    #  Focus: neutrals that look positive/negative, soft misses, mild beats
    # ══════════════════════════════════════════════════════════════════════════

    # ── HARD NEUTRAL — sounds positive but is actually neutral ────────────────
    ("HDFC Bank quarterly profit grows 3 percent year on year in line with estimates", 1),
    ("Infosys revenue grows 3.2 percent constant currency in line with guidance", 1),
    ("SBI net profit at 17000 crore meets analyst consensus no surprise", 1),
    ("TCS Q4 margin at 24.5 percent in line with expectations no beat no miss", 1),
    ("Bajaj Finance AUM grows 26 percent as guided management reiterates outlook", 1),
    ("Bank NII growth of 10 percent meets street forecast no revision needed", 1),
    ("Wipro Q1 revenue guidance of 1 to 3 percent in line with consensus view", 1),
    ("Nifty 50 recovers 200 points from lows closes flat for the week", 1),
    ("ICICI Bank loan growth of 16 percent broadly in line NIM stable", 1),
    ("Asian Paints volume growth 4 percent meets expectations no upside surprise", 1),
    ("Company delivers results broadly in line with analyst forecast no re-rating", 1),
    ("Auto sector sales for April broadly stable no major deviation from average", 1),
    ("Reliance Jio subscriber additions steady at 5 million in line with estimates", 1),
    ("HDFC Bank stock rises 1 percent after in-line results no major re-rating", 1),
    ("Management maintains FY27 guidance no change to prior forecast", 1),
    ("HUL Q4 volume growth of 3 percent meets consensus no earnings revision", 1),
    ("Pharma company EBITDA margin at 22 percent in line with 5 year average", 1),
    ("Bank provisioning at 2500 crore matches estimate no change to coverage", 1),
    ("Titan Company revenue growth of 18 percent meets but does not beat forecast", 1),
    ("Nifty Bank closes up 0.3 percent on light volume no institutional direction", 1),

    # ── HARD NEUTRAL — sounds negative but is actually neutral ────────────────
    ("HDFC Bank stock falls 2 percent on profit booking after 20 percent rally", 1),
    ("IT stocks slip 1 percent as US market gives mixed cues no fundamental change", 1),
    ("Banking index corrects 0.8 percent ahead of RBI policy no new catalyst", 1),
    ("Infosys down 1.5 percent today along with broader market decline sector neutral", 1),
    ("Sensex falls 300 points as traders book profits ahead of F&O expiry", 1),
    ("Auto stocks under mild pressure after higher than expected inventory data", 1),
    ("FMCG index marginally negative rural recovery slower than feared but stable", 1),
    ("SBI slips 1 percent on high volume in profit booking no news driven fall", 1),
    ("Market corrects after four sessions of gains no fresh negative trigger", 1),
    ("Nifty trims gains in second half corrects from intraday high by 100 points", 1),
    ("Reliance Industries falls 0.7 percent in sympathy with oil price dip", 1),
    ("Banking sector down 0.5 percent as US bank shares retreat no India specific news", 1),
    ("TCS dips 1.2 percent as investors rotate out of IT into rate sensitives", 1),
    ("Smallcap index declines 0.9 percent after strong run risk off positioning", 1),
    ("HDFC Bank slightly underperforms benchmark today no specific negative catalyst", 1),

    # ── HARD NEUTRAL — ambiguous analyst commentary ───────────────────────────
    ("Analyst expects HDFC Bank to deliver steady results no major upside catalyst", 1),
    ("Brokerages neutral on IT sector near term demand uncertain long term intact", 1),
    ("CLSA raises target by 3 percent but maintains neutral on valuation grounds", 1),
    ("Bank of America Merrill Lynch upgrades banking sector from underweight to neutral", 1),
    ("Nomura lowers target price by 2 percent keeps buy due to long term thesis", 1),
    ("Multiple brokerages keep hold with targets ranging from 1600 to 1850 no consensus", 1),
    ("Analyst sees limited upside at current valuation recommends switching to peers", 1),
    ("JPMorgan neutral on Indian banking sector macro tailwinds offset by NIM pressure", 1),
    ("Research note: HDFC Bank fairly valued risk reward balanced at current levels", 1),
    ("Brokerages divided on FMCG outlook half buy half hold no clear re-rating", 1),

    # ── HARD NEUTRAL — results with mixed components ──────────────────────────
    ("HDFC Bank profit beats but net interest margin misses net impact neutral", 1),
    ("Infosys revenue beats but margins compress by 50 basis points net mixed", 1),
    ("Auto company volumes beat but realizations fall price and volume offset", 1),
    ("FMCG company margin beats but volume growth disappoints net result neutral", 1),
    ("IT company wins large deal but loses another client of similar size", 1),
    ("Bank asset quality improves but fee income misses estimates total result mixed", 1),
    ("Pharma company domestic business strong but US revenue decline offsets", 1),
    ("Revenue beats estimates but EBITDA misses due to unexpected one-time cost", 1),
    ("Strong top line but sharp increase in other income masks operating weakness", 1),
    ("Capital goods company order inflows beat but margins compress execution pressure", 1),

    # ── HARD NEUTRAL — news from social media / clickbait titles ─────────────
    ("HDFC Bank stocks crash as Nifty corrects all you need to know", 1),
    ("5 reasons why banking stocks are volatile today experts weigh in", 1),
    ("Should you buy HDFC Bank at current levels analysts debate target price", 1),
    ("HDFC Bank near 52 week high is it time to buy sell or hold", 1),
    ("What to do with Infosys stock after Q4 results buy hold or sell", 1),
    ("Stocks that moved most today HDFC Bank SBI TCS among big movers", 1),
    ("Market expert sees potential for 10 percent move either way in Nifty", 1),
    ("HDFC Bank chart analysis key support resistance levels updated", 1),
    ("Nifty likely to be volatile this week here is what traders say", 1),
    ("Stock market today why your portfolio may be down despite Nifty flat", 1),

    # ── HARD NEGATIVE — soft misses that are genuinely negative ──────────────
    ("HDFC Bank Q4 profit rises only 2 percent misses consensus estimate of 8 percent", 0),
    ("Infosys Q1 revenue grows 1.8 percent constant currency below lower end of guidance", 0),
    ("SBI NIM falls 10 basis points below estimates rising deposit cost pressure", 0),
    ("TCS headcount flat for second consecutive quarter demand recovery further delayed", 0),
    ("Wipro issues Q2 guidance below street mid-point at 1 percent constant currency", 0),
    ("Auto OEM wholesale dispatches fall 4 percent on channel inventory correction", 0),
    ("HDFC Bank part time chairman appointment rejected by shareholders governance concern", 0),
    ("FMCG company volume growth slows to 1 percent rural demand weaker than expected", 0),
    ("Bajaj Finance gross NPA ticks up 10 basis points credit cost guidance raised", 0),
    ("L&T order inflow growth slows to 8 percent below estimates of 15 percent", 0),
    ("Bank gross slippage rate increases to 1.5 percent from 1.2 percent quarter ago", 0),
    ("IT sector deal signings fall 20 percent sequentially slowest in six quarters", 0),
    ("RBI raises concern over rapid growth in personal loans systemic risk flagged", 0),
    ("Reliance Retail revenue misses estimates footfall growth moderating", 0),
    ("ONGC Q4 profits fall 12 percent on higher statutory levies and lower realization", 0),
    ("NSE bans 5 brokers for client fund misuse market confidence dented", 0),
    ("Pharma company US product recall voluntary action revenue impact significant", 0),
    ("Tata Steel reports surprise loss on European operations higher than expected impairment", 0),
    ("Company management gives cautious commentary on demand environment visibility poor", 0),
    ("Export-oriented company warns of order cancellations as US tariff risk rises", 0),

    # ── HARD NEGATIVE — macro / systemic negatives ────────────────────────────
    ("India bond market selloff 10 year yield rises 20 basis points liquidity tightens", 0),
    ("RBI reduces banking system liquidity as inflation surprises on the upside", 0),
    ("Credit rating agency changes India outlook to negative from stable deficit concern", 0),
    ("India industrial output IIP contracts for second month manufacturing slumps", 0),
    ("India composite PMI falls to 54.7 from 58.1 pace of expansion slowing", 0),
    ("CPI inflation re-accelerates to 5.8 percent RBI rate cut hopes pushed out", 0),
    ("India rupee falls past 85 to fresh record low capital outflow intensifies", 0),
    ("FII outflows cross 50000 crore in one month record in history for India", 0),
    ("Nifty 50 enters correction territory down 10 percent from recent peak", 0),
    ("SEBI tightens margin requirements futures trading cost rises speculation falls", 0),

    # ── HARD NEGATIVE — sector-specific subtle negatives ─────────────────────
    ("Bank credit card spends growth decelerates to 8 percent from 22 percent prior", 0),
    ("IT sector pricing pressure intensifies in application maintenance contracts", 0),
    ("FMCG company loses market share to regional peers in key volume segment", 0),
    ("Specialty chemical company customer destocking impact persists into Q2", 0),
    ("Auto ancillary company faces EV disruption risk traditional part volumes falling", 0),
    ("Real estate developer faces RERA compliance issues project launches halted", 0),
    ("Hospital chain faces regulatory investigation overcharging complaint", 0),
    ("Airline company fuel hedging loss amplifies quarterly operating loss", 0),
    ("Capital goods company working capital cycle elongates receivables spike", 0),
    ("Telecom company loses 2 million subscribers to Jio on tariff differential", 0),

    # ── HARD POSITIVE — modest beats that are genuinely positive ──────────────
    ("HDFC Bank NIM surprises positively at 3.52 percent beat of 8 basis points", 2),
    ("Infosys Q4 constant currency growth of 5.4 percent above guidance of 3 to 5", 2),
    ("SBI credit cost falls to 0.5 percent best in 10 years asset quality pristine", 2),
    ("TCS deal wins of 9.4 billion dollars highest ever provides strong revenue cover", 2),
    ("Wipro beats Q3 IT services revenue by 60 million dollars strong execution", 2),
    ("Bajaj Finance raises AUM guidance for third consecutive quarter conviction high", 2),
    ("ICICI Bank return on equity crosses 18 percent best among private sector banks", 2),
    ("Auto company gains 150 basis points market share in SUV segment premium mix up", 2),
    ("FMCG company surprises with 9 percent volume growth rural market inflecting", 2),
    ("IT company attrition falls to 8.2 percent lowest in 5 years retention improves", 2),
    ("Bank CASA ratio rises to 46 percent sequential improvement in low cost deposits", 2),
    ("Pharma company first ever FDA approval for complex peptide drug breakthrough", 2),
    ("Company announces 25 percent increase in dividend payout ratio higher return", 2),
    ("Credit card spends growth re-accelerates to 20 percent post festive recovery", 2),
    ("Infrastructure company cash flow positive for first time in 4 years turning point", 2),
    ("Cement company EBITDA per tonne hits record on cost efficiencies pricing stable", 2),
    ("Insurance company new business premium growth of 28 percent beats sector growth", 2),
    ("Consumer durable company launches premium product gaining mix benefit margins up", 2),
    ("NBFC NIM stable despite rate cycle company raises FY27 growth guidance", 2),
    ("Company reduces net debt by 5000 crore ahead of schedule leverage normalising", 2),

    # ── HARD POSITIVE — institutional flows and policy tailwinds ──────────────
    ("SEBI allows higher FII limits in banking stocks 3 billion dollar headroom created", 2),
    ("RBI announces measures to improve banking system liquidity lending improves", 2),
    ("Government raises FDI limit in insurance sector to 100 percent sector positive", 2),
    ("PLI scheme disbursements accelerate capex in electronics pharma auto benefits", 2),
    ("Budget allocates record 11 lakh crore for infrastructure spending biggest ever", 2),
    ("India added to Bloomberg EM bond index 5 billion dollar passive inflow expected", 2),
    ("MSCI increases India EM weight positive for passive FII inflow of 2 billion", 2),
    ("Sovereign wealth funds increase India allocation 3 large GIC ADIA CDPQ invest", 2),
    ("Mutual fund industry AUM crosses 60 lakh crore milestone SIP flows record high", 2),
    ("India receives investment grade upgrade from DBRS Morningstar cost of capital falls", 2),

    # ── HARD POSITIVE — specific stock re-rating triggers ─────────────────────
    ("HDFC Bank RBI lifts all restrictions CEO says execution on track", 2),
    ("Infosys wins Siemens multi-year transformation deal worth 1.8 billion dollars", 2),
    ("SBI announces 1 for 10 bonus issue first in 5 years strong signal", 2),
    ("Tata Motors JLR order book rises to 6 month high EV demand inflects", 2),
    ("Reliance Industries Jio financial services IPO announced at premium valuation", 2),
    ("Apollo Hospitals included in Nifty 50 healthcare weight significant", 2),
    ("Kotak Bank founder promoter increases shareholding open market positive signal", 2),
    ("NTPC net zero carbon target 2047 ESG re-rating catalyses green premium", 2),
    ("Bajaj Auto premium motorcycle segment crosses 50 percent revenue share milestone", 2),
    ("Dixon Technologies Samsung partnership extended secures 3 year revenue visibility", 2),

    # ══════════════════════════════════════════════════════════════════════════
    #  BATCH 4 — 60 targeted hard negatives (fix 78% recall gap)
    #  These are subtle bearish headlines the model confuses with neutral
    # ══════════════════════════════════════════════════════════════════════════

    # Guidance cuts framed diplomatically
    ("Management guides conservatively for next quarter citing demand uncertainty", 0),
    ("Company withdraws full year guidance citing visibility concerns macro uncertain", 0),
    ("CEO acknowledges challenges says recovery will take longer than expected", 0),
    ("CFO warns of near term headwinds margins unlikely to recover before H2", 0),
    ("Management says demand environment remains challenging recovery pushed to FY28", 0),
    ("Company trims revenue growth guidance from 15 to 10 percent citing slowdown", 0),
    ("Analyst day disappoints management lowers medium term EBITDA margin target", 0),
    ("Company revises down capex guidance signals growth expectations lowered", 0),

    # Soft misses framed in positive language (tricky for model)
    ("HDFC Bank reports profit growth slightly below street due to higher provisions", 0),
    ("IT company revenue at lower end of guidance no upside catalyst visible", 0),
    ("Bank credit growth disappoints at 11 percent versus prior quarter 16 percent", 0),
    ("Auto company reports volume growth but lower than both quarter ago and year ago", 0),
    ("Pharma company USFDA filing pipeline dries up no new product launches FY27", 0),
    ("Profit growth of 5 percent reported below estimated 13 percent stock falls", 0),
    ("Company EBITDA per unit declines for third consecutive quarter pricing erodes", 0),
    ("Insurance company embedded value growth disappoints street below estimates", 0),

    # Subtle asset quality stress (banking sector)
    ("Bank restructured book increases 30 percent sequentially future NPA risk", 0),
    ("Bank special mention accounts SMA-2 increase 25 percent stress building", 0),
    ("Microfinance loans to stressed borrowers increase NPA emergence risk rises", 0),
    ("Bank unsecured retail portfolio grows faster than secured risk concentration", 0),
    ("Watchlist accounts added back as downgrade risk elevated", 0),
    ("Credit bureau data shows overleverage in personal loan segment banks exposed", 0),
    ("RBI annual inspection report flags weaknesses in internal audit governance", 0),

    # Macro signals that are genuinely negative
    ("India manufacturing PMI falls from 57 to 54.3 pace of expansion slowing", 0),
    ("Services sector PMI slips to 56.5 from 58.1 previous month momentum fades", 0),
    ("India core inflation rises 30 basis points RBI rate cut pushed to December", 0),
    ("India goods exports contract 4 percent for second consecutive month", 0),
    ("Foreign exchange outflows from equities and bonds combined hit 70000 crore", 0),
    ("Advance tax collections below budget estimate government revenue shortfall risk", 0),
    ("India external debt to GDP rises above 20 percent first time in 5 years", 0),
    ("RBI OMO open market operations signal liquidity remains tight for extended period", 0),

    # Corporate deterioration signals
    ("Company free cash flow negative for three consecutive quarters debt rising", 0),
    ("Promoter share pledge increases to 55 percent from 40 percent financial stress", 0),
    ("Company misses debt repayment timeline seeks extension from lenders", 0),
    ("Conglomerate announces sale of core business asset to repay debt distress", 0),
    ("Company announces emergency rights issue at 30 percent discount to fund losses", 0),
    ("Subsidiary reports loss management says parent support likely balance sheet risk", 0),
    ("Company accounts under scrutiny after auditor flags going concern doubt", 0),

    # Sector deterioration
    ("Airline fuel costs rise 18 percent quarterly yield flat profitability at risk", 0),
    ("Hospital chain sees fall in international patient volumes forex headwind", 0),
    ("Real estate developer delays project delivery third time buyers filing complaints", 0),
    ("Textile company order book shrinks as fast fashion brands cut order quantities", 0),
    ("Specialty chemical client inventory destocking persists beyond initial estimate", 0),
    ("Two wheeler company dealer inventory at 9 weeks highest in two years pressure", 0),
    ("Chemical company facing pricing pressure as China volumes flood India market", 0),

    # Technical negative signals (softer language that should still be negative)
    ("Nifty forms lower high lower low pattern on daily chart trend turning negative", 0),
    ("Stock underperforms broader market for 10 consecutive sessions relative weakness", 0),
    ("Foreign investors sell banking stocks for 8 consecutive sessions sustained exit", 0),
    ("Index heavyweight stock breaks 200 day moving average on above average volume", 0),
    ("India VIX rises from 14 to 18 over 5 sessions fear index creeping up", 0),
    ("Advance decline ratio worsens 3 declines for every advance broad weakness", 0),
    ("Stock hits new 52 week low despite broader market recovery extreme weakness", 0),

    # Macro global negative spillover
    ("US tariffs on Indian goods announced impact on IT exports and pharma assessed", 0),
    ("Oil prices spike 12 percent on supply cut India import bill rises sharply", 0),
    ("Emerging market currencies under pressure dollar strength hits rupee INR falls", 0),
    ("Global risk off intensifies India VIX jumps Nifty futures in discount", 0),

]


def get_dataset():
    texts  = [t for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]
    return texts, labels


def get_label_counts():
    from collections import Counter
    return Counter(l for _, l in TRAINING_DATA)


if __name__ == "__main__":
    texts, labels = get_dataset()
    counts = get_label_counts()
    print(f"Total samples : {len(texts)}")
    print(f"Negative (0)  : {counts[0]}")
    print(f"Neutral  (1)  : {counts[1]}")
    print(f"Positive (2)  : {counts[2]}")
