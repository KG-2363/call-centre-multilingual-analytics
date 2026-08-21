"""Conversation template bank — the seed corpus.

Openly-licensed *Indian* contact-centre audio with code-switching barely exists;
IndicVoices and Shrutilipi are read speech, and Common Voice hi is single-speaker
prompts. None of them contain agent/customer turn structure, hold events or
compliance language. So rather than pretend a read-speech corpus is call data,
this bank encodes 15 hand-written scenarios drawn from the intents that actually
dominate Indian telecom/fintech support queues, each written in one of four
language modes:

    en        pure English (metro, English-preferring customer)
    hi        Devanagari Hindi
    hinglish  intra-sentential mixing, Latin script  <- the hard, common case
    hi_latn   monolingual Hindi typed/spoken in Latin script

`ccma seed` expands these deterministically into 30 calls with varied agents,
timestamps, hold behaviour and injected ASR-style noise. Because the ground-truth
text is known, this corpus doubles as the **evaluation set**: it is the only
place in the system where true transcripts and true labels exist, which is what
makes the calibration in ccma.quality possible.

To swap in real audio, implement a loader that emits CallMetadata + audio files
and point `ccma ingest` at it — nothing downstream depends on this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models.enums import ComplianceRule, ResolutionStatus, SpeakerRole


@dataclass(frozen=True)
class TurnTemplate:
    role: SpeakerRole
    text: str
    lang: str
    ms: int = 3000
    # Marks a turn the customer speaks over the agent — used to generate
    # realistic overlapping-speech and diarization-error test cases.
    overlaps_next: bool = False


@dataclass(frozen=True)
class CallTemplate:
    template_id: str
    intent_l1: str
    intent_l2: str
    intent_l3: str | None
    language_style: str
    queue: str
    resolution: ResolutionStatus
    escalated: bool
    fcr: bool
    compliance_present: list[ComplianceRule]
    turns: list[TurnTemplate] = field(default_factory=list)


A = SpeakerRole.AGENT
C = SpeakerRole.CUSTOMER

# TTS voice assignment per language mode (edge-tts short names).
TTS_VOICES: dict[str, dict[str, str]] = {
    "en": {"AGENT": "en-IN-PrabhatNeural", "CUSTOMER": "en-IN-NeerjaNeural"},
    "hi": {"AGENT": "hi-IN-MadhurNeural", "CUSTOMER": "hi-IN-SwaraNeural"},
    "hinglish": {"AGENT": "hi-IN-MadhurNeural", "CUSTOMER": "en-IN-NeerjaNeural"},
    "hi_latn": {"AGENT": "hi-IN-MadhurNeural", "CUSTOMER": "hi-IN-SwaraNeural"},
}

DISCLOSURE = {
    "en": "Thank you for calling Bharat Telecom. This call is being recorded for quality and training purposes.",
    "hi": "भारत टेलीकॉम में आपका स्वागत है। गुणवत्ता और प्रशिक्षण के लिए यह कॉल रिकॉर्ड की जा रही है।",
    "hinglish": "Bharat Telecom mein aapka swagat hai. Quality purposes ke liye yeh call record ki ja rahi hai.",
    "hi_latn": "Bharat Telecom mein aapka swagat hai. Quality aur training ke liye yeh call record ki ja rahi hai.",
}

TEMPLATES: list[CallTemplate] = [
    # ---------------------------------------------------------------- 1
    CallTemplate(
        "T01_billing_overcharge_hinglish",
        "BILLING", "OVERCHARGE_DISPUTE", "UNEXPECTED_DATA_CHARGE",
        "hinglish", "billing_hi", ResolutionStatus.RESOLVED, False, True,
        [ComplianceRule.RECORDING_DISCLOSURE, ComplianceRule.IDENTITY_VERIFICATION,
         ComplianceRule.PROFESSIONAL_CLOSING],
        [
            TurnTemplate(A, DISCLOSURE["hinglish"], "hinglish", 6000),
            TurnTemplate(A, "Main Rohit bol raha hoon, how may I help you today?", "hinglish", 3200),
            TurnTemplate(C, "Haan sir, mera bill is month bahut zyada aa gaya hai. Normally 399 ka plan hai lekin 847 rupaye kat gaye.", "hinglish", 8500),
            TurnTemplate(A, "I understand your concern sir. Verification ke liye aap apna registered mobile number aur date of birth bata sakte hain?", "hinglish", 6500),
            TurnTemplate(C, "Haan, number 98XX XX1234 hai aur DOB pandrah August unnees sau nabbe.", "hinglish", 5500),
            TurnTemplate(A, "Thank you for verifying. Ek minute hold kariye, main aapka bill breakdown check karta hoon.", "hinglish", 5000),
            TurnTemplate(A, "Sir, I can see 448 rupees ka extra data charge hai. Aapka daily limit exceed ho gaya tha teen din.", "hinglish", 7000),
            TurnTemplate(C, "Lekin mujhe koi notification nahi aaya tha! Yeh galat hai, mujhe pata hi nahi chala.", "hinglish", 6000),
            TurnTemplate(A, "Aap bilkul sahi keh rahe hain sir. Notification failure hamari taraf se issue tha. Main 448 rupaye ka waiver process kar deta hoon.", "hinglish", 8000),
            TurnTemplate(C, "Achha theek hai, thank you. Kitne din mein reflect hoga?", "hinglish", 4000),
            TurnTemplate(A, "Next billing cycle mein credit ho jayega, forty-eight hours ke andar SMS confirmation aa jayega.", "hinglish", 5500),
            TurnTemplate(C, "Perfect, bahut dhanyavaad.", "hinglish", 2500),
            TurnTemplate(A, "Thank you for calling Bharat Telecom. Have a great day sir!", "hinglish", 3500),
        ],
    ),
    # ---------------------------------------------------------------- 2
    CallTemplate(
        "T02_network_outage_hi",
        "TECHNICAL_SUPPORT", "NETWORK_CONNECTIVITY", "NO_SIGNAL_AREA",
        "hi", "tech_hi", ResolutionStatus.PARTIALLY_RESOLVED, False, False,
        [ComplianceRule.RECORDING_DISCLOSURE, ComplianceRule.PROFESSIONAL_CLOSING],
        [
            TurnTemplate(A, DISCLOSURE["hi"], "hi", 6500),
            TurnTemplate(A, "नमस्ते, मैं प्रिया बोल रही हूँ। मैं आपकी क्या सहायता कर सकती हूँ?", "hi", 4000),
            TurnTemplate(C, "मैडम, पिछले तीन दिन से मेरे इलाके में बिल्कुल नेटवर्क नहीं आ रहा है। कोई सिग्नल ही नहीं है।", "hi", 7500),
            TurnTemplate(A, "मुझे खेद है कि आपको परेशानी हो रही है। कृपया अपना पिन कोड बताइए।", "hi", 5000),
            TurnTemplate(C, "पाँच शून्य शून्य शून्य आठ एक, हैदराबाद।", "hi", 3500),
            TurnTemplate(A, "एक मिनट रुकिए, मैं जाँच करती हूँ।", "hi", 3000),
            TurnTemplate(A, "सर, आपके क्षेत्र में टावर का रखरखाव चल रहा है। यह काम अगले दो दिन में पूरा हो जाएगा।", "hi", 7000),
            TurnTemplate(C, "दो दिन और? मेरा काम पूरी तरह रुका हुआ है। मुझे ऑनलाइन क्लास लेनी होती है।", "hi", 6500),
            TurnTemplate(A, "मैं समझती हूँ। मैं आपके खाते में तीन दिन का बिल माफ़ कर देती हूँ और प्राथमिकता शिकायत दर्ज कर रही हूँ।", "hi", 8500),
            TurnTemplate(C, "ठीक है, लेकिन कृपया जल्दी करवाइए।", "hi", 3500),
            TurnTemplate(A, "ज़रूर सर। शिकायत संख्या है नौ चार दो सात एक। धन्यवाद, आपका दिन शुभ हो।", "hi", 6000),
        ],
    ),
    # ---------------------------------------------------------------- 3
    CallTemplate(
        "T03_payment_failed_escalation_hinglish",
        "BILLING", "PAYMENT_FAILURE", "AMOUNT_DEBITED_NOT_CREDITED",
        "hinglish", "billing_hi", ResolutionStatus.ESCALATED, True, False,
        [ComplianceRule.RECORDING_DISCLOSURE, ComplianceRule.IDENTITY_VERIFICATION],
        [
            TurnTemplate(A, DISCLOSURE["hinglish"], "hinglish", 6000),
            TurnTemplate(C, "Dekhiye main teesri baar call kar raha hoon! Mere account se 1200 rupaye kat gaye hain aur recharge hua hi nahi.", "hinglish", 9000, overlaps_next=True),
            TurnTemplate(A, "Sir main samajh sakta hoon, aap please thoda calm ho jaiye, main abhi check karta hoon.", "hinglish", 5500),
            TurnTemplate(C, "Calm? Chaar din ho gaye hain! Har baar bolte ho 24 hours mein solve hoga. Yeh kya mazaak hai?", "hinglish", 8000),
            TurnTemplate(A, "Aapki registered mobile number bata dijiye verification ke liye.", "hinglish", 4000),
            TurnTemplate(C, "97XX XX8876. Aur haan, agar aaj solve nahi hua to main consumer forum mein complaint karunga.", "hinglish", 7500),
            TurnTemplate(A, "Sir, main dekh raha hoon ki aapka transaction bank ke paas pending state mein hai. Yeh hamare system mein reflect nahi hua.", "hinglish", 8500),
            TurnTemplate(C, "To ab main kya karoon? Paisa mera gaya, aur aap bol rahe ho bank ka issue hai. Mujhe aapke senior se baat karni hai.", "hinglish", 9000),
            TurnTemplate(A, "Bilkul sir, main aapko apne floor supervisor se connect karta hoon. Please hold kariye.", "hinglish", 5500),
            TurnTemplate(C, "Theek hai, lekin jaldi. Main aur wait nahi kar sakta.", "hinglish", 4000),
        ],
    ),
    # ---------------------------------------------------------------- 4
    CallTemplate(
        "T04_refund_status_en",
        "BILLING", "REFUND_STATUS", "REFUND_DELAY",
        "en", "billing_en", ResolutionStatus.RESOLVED, True, True,
        [ComplianceRule.RECORDING_DISCLOSURE, ComplianceRule.IDENTITY_VERIFICATION,
         ComplianceRule.PROFESSIONAL_CLOSING],
        [
            TurnTemplate(A, DISCLOSURE["en"], "en", 6000),
            TurnTemplate(A, "Good afternoon, my name is Anjali. How may I assist you?", "en", 3500),
            TurnTemplate(C, "Hi, I cancelled my broadband plan two weeks ago and I was told the refund would come in seven working days. It still hasn't arrived.", "en", 9000),
            TurnTemplate(A, "I'm sorry about the delay. Could you confirm your account number and the email registered with us?", "en", 5500),
            TurnTemplate(C, "Account is BT four four nine two one, and the email is customer at example dot com.", "en", 5000),
            TurnTemplate(A, "Thank you. Let me pull up the refund record, please stay on the line.", "en", 4000),
            TurnTemplate(A, "I can see the refund of 2,340 rupees was initiated on the eighteenth but it failed because the account was closed at your bank.", "en", 8000),
            TurnTemplate(C, "Oh, that's right, I switched banks last month. Nobody informed me it had failed though.", "en", 6000),
            TurnTemplate(A, "That's a valid point and I apologise for that. I'll update your new account details and re-initiate it right away.", "en", 7000),
            TurnTemplate(C, "Great. The new account is with HDFC, number ending three three one two.", "en", 5000),
            TurnTemplate(A, "Updated. The refund will reflect within three working days and you'll get an SMS confirmation.", "en", 6000),
            TurnTemplate(C, "Thank you, that's very helpful.", "en", 2500),
            TurnTemplate(A, "My pleasure. Thank you for calling Bharat Telecom, have a good day.", "en", 4000),
        ],
    ),
    # ---------------------------------------------------------------- 5
    CallTemplate(
        "T05_fraud_unauthorized_hi_latn",
        "FRAUD_SECURITY", "UNAUTHORIZED_TRANSACTION", "SIM_SWAP_SUSPECTED",
        "hi_latn", "fraud", ResolutionStatus.ESCALATED, True, False,
        [ComplianceRule.RECORDING_DISCLOSURE, ComplianceRule.IDENTITY_VERIFICATION,
         ComplianceRule.NO_FULL_CARD_READBACK, ComplianceRule.DPDP_CONSENT],
        [
            TurnTemplate(A, DISCLOSURE["hi_latn"], "hi_latn", 6500),
            TurnTemplate(C, "Sir meri SIM achanak band ho gayi aur mere bank account se paise nikal gaye. Mujhe lagta hai kisi ne SIM swap kiya hai.", "hi_latn", 9500),
            TurnTemplate(A, "Yeh bahut gambhir maamla hai. Main turant aapki SIM block kar raha hoon. Aap apna naam aur pata bataiye.", "hi_latn", 7500),
            TurnTemplate(C, "Mera naam Suresh Kumar hai, pata hai sector pandrah, Noida.", "hi_latn", 5000),
            TurnTemplate(A, "Dhanyavaad. Kripya apna poora card number mat bataiye, sirf aakhri chaar ank bataiye.", "hi_latn", 6500),
            TurnTemplate(C, "Aakhri chaar ank hain saat do naau ek.", "hi_latn", 3500),
            TurnTemplate(A, "Theek hai. Aapki jaankari hum fraud investigation team ke saath share karenge, kya aapki iske liye sahmati hai?", "hi_latn", 7500),
            TurnTemplate(C, "Haan bilkul, jo karna hai jaldi kariye. Mere paas sirf yehi account hai.", "hi_latn", 5500),
            TurnTemplate(A, "SIM block ho gayi hai. Main case fraud team ko escalate kar raha hoon, wo chaubees ghante mein aapko call karenge.", "hi_latn", 8000),
            TurnTemplate(C, "Chaubees ghante bahut zyada hai lekin theek hai. Please follow up zaroor kariyega.", "hi_latn", 6000),
        ],
    ),
    # ---------------------------------------------------------------- 6
    CallTemplate(
        "T06_plan_upsell_hinglish",
        "SALES", "PLAN_UPGRADE", "DATA_ADDON",
        "hinglish", "sales", ResolutionStatus.RESOLVED, False, True,
        [ComplianceRule.RECORDING_DISCLOSURE, ComplianceRule.MIS_SELLING_DISCLAIMER,
         ComplianceRule.PROFESSIONAL_CLOSING],
        [
            TurnTemplate(A, DISCLOSURE["hinglish"], "hinglish", 6000),
            TurnTemplate(A, "Sir, aapke current usage pattern ke basis par humare paas ek better plan hai. Interested honge?", "hinglish", 6000),
            TurnTemplate(C, "Batayiye kya hai. Lekin main abhi 399 wala use kar raha hoon.", "hinglish", 4500),
            TurnTemplate(A, "499 ke plan mein aapko daily 3GB milega instead of 1.5GB, plus OTT subscription free hai.", "hinglish", 7000),
            TurnTemplate(C, "Hmm, 100 rupaye extra. Koi hidden charges to nahi hain?", "hinglish", 4500),
            TurnTemplate(A, "Nahi sir, koi hidden charge nahi hai. Yeh price GST inclusive hai. Plan renewal automatic nahi hoga, aap kabhi bhi change kar sakte hain.", "hinglish", 8500),
            TurnTemplate(C, "Aur agar mujhe pasand nahi aaya to?", "hinglish", 3000),
            TurnTemplate(A, "Aap next cycle mein wapas apne purane plan par ja sakte hain, koi penalty nahi hai.", "hinglish", 5500),
            TurnTemplate(C, "Theek hai phir, activate kar dijiye.", "hinglish", 3000),
            TurnTemplate(A, "Done sir. Aapka naya plan aaj midnight se effective ho jayega. Thank you for choosing Bharat Telecom!", "hinglish", 6500),
        ],
    ),
    # ---------------------------------------------------------------- 7
    CallTemplate(
        "T07_dnd_violation_hi",
        "COMPLAINT", "DND_VIOLATION", "PROMOTIONAL_SPAM",
        "hi", "complaint", ResolutionStatus.RESOLVED, False, True,
        [ComplianceRule.RECORDING_DISCLOSURE, ComplianceRule.DND_ACKNOWLEDGEMENT,
         ComplianceRule.GRIEVANCE_REDRESSAL_INFO],
        [
            TurnTemplate(A, DISCLOSURE["hi"], "hi", 6500),
            TurnTemplate(C, "मैंने डी एन डी चालू करवाया था फिर भी मुझे रोज़ दस प्रचार संदेश आते हैं। यह बहुत परेशान करने वाला है।", "hi", 9000),
            TurnTemplate(A, "मुझे इसके लिए खेद है। मैं आपकी डी एन डी स्थिति जाँच रही हूँ, कृपया प्रतीक्षा कीजिए।", "hi", 6000),
            TurnTemplate(A, "सर, आपकी डी एन डी पंजीकरण आंशिक है, केवल बैंकिंग श्रेणी के लिए सक्रिय है।", "hi", 7000),
            TurnTemplate(C, "मैंने तो पूरी तरह बंद करने को कहा था। यह गलती किसकी है?", "hi", 5500),
            TurnTemplate(A, "यह हमारी प्रणाली की त्रुटि हो सकती है। मैं अभी सभी श्रेणियों के लिए पूर्ण डी एन डी सक्रिय कर रही हूँ।", "hi", 8000),
            TurnTemplate(C, "ठीक है। कितने समय में लागू होगा?", "hi", 3500),
            TurnTemplate(A, "सात दिन के भीतर पूरी तरह लागू हो जाएगा। यदि फिर भी संदेश आएँ तो आप ट्राई के शिकायत पोर्टल पर भी दर्ज करा सकते हैं।", "hi", 9000),
            TurnTemplate(C, "अच्छा, जानकारी के लिए धन्यवाद।", "hi", 3000),
            TurnTemplate(A, "धन्यवाद सर, आपका दिन मंगलमय हो।", "hi", 3500),
        ],
    ),
    # ---------------------------------------------------------------- 8
    CallTemplate(
        "T08_delivery_delay_hinglish",
        "LOGISTICS", "DELIVERY_DELAY", "SIM_NOT_DELIVERED",
        "hinglish", "logistics", ResolutionStatus.CALLBACK_SCHEDULED, False, False,
        [ComplianceRule.RECORDING_DISCLOSURE, ComplianceRule.PROFESSIONAL_CLOSING],
        [
            TurnTemplate(A, DISCLOSURE["hinglish"], "hinglish", 6000),
            TurnTemplate(C, "Maine ek week pehle new SIM order kiya tha, abhi tak delivery nahi hui. Tracking bhi update nahi ho raha.", "hinglish", 8000),
            TurnTemplate(A, "Sorry for the inconvenience ma'am. Aap order ID bata sakti hain?", "hinglish", 4500),
            TurnTemplate(C, "Order ID hai BT double seven three one nine.", "hinglish", 4000),
            TurnTemplate(A, "Ek second hold kariye... Ma'am, aapka shipment Bengaluru hub mein stuck hai, courier partner ki taraf se delay hai.", "hinglish", 8500),
            TurnTemplate(C, "Yeh to bahut kharab service hai. Mujhe SIM urgently chahiye tha office ke liye.", "hinglish", 6000),
            TurnTemplate(A, "Main courier team ko escalate kar rahi hoon. Aapko kal shaam tak update call aayega.", "hinglish", 6000),
            TurnTemplate(C, "Kal shaam? Aur agar tab bhi nahi aaya to?", "hinglish", 3500),
            TurnTemplate(A, "Us case mein hum fresh SIM express delivery se bhej denge, no extra charge. Main callback schedule kar rahi hoon.", "hinglish", 8000),
            TurnTemplate(C, "Theek hai, dekhte hain.", "hinglish", 2500),
            TurnTemplate(A, "Thank you for your patience ma'am. Have a good day.", "hinglish", 3500),
        ],
    ),
    # ---------------------------------------------------------------- 9
    CallTemplate(
        "T09_retention_churn_hinglish",
        "RETENTION", "CANCELLATION_REQUEST", "COMPETITOR_OFFER",
        "hinglish", "retention", ResolutionStatus.RESOLVED, False, True,
        [ComplianceRule.RECORDING_DISCLOSURE, ComplianceRule.MIS_SELLING_DISCLAIMER,
         ComplianceRule.PROFESSIONAL_CLOSING],
        [
            TurnTemplate(A, DISCLOSURE["hinglish"], "hinglish", 6000),
            TurnTemplate(C, "Main apna connection band karwana chahta hoon. Jio mujhe same plan half price mein de raha hai.", "hinglish", 7500),
            TurnTemplate(A, "Sir aapko jaate hue dekhna bahut dukh ki baat hai. Aap humare saath teen saal se hain. Main kuch check kar sakta hoon?", "hinglish", 8000),
            TurnTemplate(C, "Dekh lijiye, lekin mera mann bana hua hai. Network bhi thoda weak rehta hai aajkal.", "hinglish", 6500),
            TurnTemplate(A, "Sir, loyalty customer ke liye main aapko 30% discount de sakta hoon next six months ke liye, aur ek network booster bhi install karwa sakta hoon free mein.", "hinglish", 10000),
            TurnTemplate(C, "Booster free mein? Uska koi installation charge to nahi?", "hinglish", 4000),
            TurnTemplate(A, "Bilkul free hai sir, koi hidden cost nahi. Discount ke baad aapka monthly 279 ho jayega.", "hinglish", 6500),
            TurnTemplate(C, "Hmm, yeh to theek offer hai. Chalo phir rehne dete hain.", "hinglish", 4500),
            TurnTemplate(A, "Bahut bahut dhanyavaad sir. Booster installation ke liye technician kal aapko call karega. Have a great day!", "hinglish", 7000),
        ],
    ),
    # ---------------------------------------------------------------- 10
    CallTemplate(
        "T10_kyc_document_hi_latn",
        "ACCOUNT_MANAGEMENT", "KYC_UPDATE", "DOCUMENT_REJECTED",
        "hi_latn", "account", ResolutionStatus.PARTIALLY_RESOLVED, False, False,
        [ComplianceRule.RECORDING_DISCLOSURE, ComplianceRule.IDENTITY_VERIFICATION,
         ComplianceRule.DPDP_CONSENT],
        [
            TurnTemplate(A, DISCLOSURE["hi_latn"], "hi_latn", 6500),
            TurnTemplate(C, "Maine KYC ke liye Aadhaar upload kiya tha lekin wo reject ho gaya. Kyun hua yeh?", "hi_latn", 6500),
            TurnTemplate(A, "Main check karta hoon sir. Kripya apna registered number aur janm tithi bataiye.", "hi_latn", 5500),
            TurnTemplate(C, "Number 88XX XX4409, janm tithi bees March unnees sau pichanve.", "hi_latn", 5000),
            TurnTemplate(A, "Sir, aapka document isliye reject hua kyunki photo dhundhli thi aur address wala hissa kata hua tha.", "hi_latn", 8000),
            TurnTemplate(C, "Lekin mujhe koi reason nahi bataya gaya tha! Sirf reject ka message aaya.", "hi_latn", 6000),
            TurnTemplate(A, "Aap sahi keh rahe hain, hamari taraf se communication adhoora tha. Main aapko dobara upload ka link bhej raha hoon.", "hi_latn", 8000),
            TurnTemplate(C, "Theek hai. Kya main branch mein jaakar bhi kara sakta hoon?", "hi_latn", 4500),
            TurnTemplate(A, "Haan bilkul, aap nazdeeki store par ja sakte hain. Kya hum aapka document verification partner ke saath share kar sakte hain?", "hi_latn", 7500),
            TurnTemplate(C, "Haan sahmati hai. Main kal store chala jaunga.", "hi_latn", 4000),
        ],
    ),
    # ---------------------------------------------------------------- 11
    CallTemplate(
        "T11_abusive_escalation_hinglish",
        "COMPLAINT", "SERVICE_QUALITY", "REPEATED_FAILURE",
        "hinglish", "complaint", ResolutionStatus.ESCALATED, True, False,
        [ComplianceRule.RECORDING_DISCLOSURE],
        [
            TurnTemplate(A, DISCLOSURE["hinglish"], "hinglish", 6000),
            TurnTemplate(C, "Aap log bilkul bekaar ho! Paanchvi baar call kar raha hoon, koi solution nahi milta.", "hinglish", 7000, overlaps_next=True),
            TurnTemplate(A, "Sir main aapki baat samajh raha hoon, mujhe ek chance dijiye issue solve karne ka.", "hinglish", 5500),
            TurnTemplate(C, "Chance? Ek mahina ho gaya! Mera internet chalta hi nahi hai aur bill pura aa raha hai. Yeh dhokha hai.", "hinglish", 8500),
            TurnTemplate(A, "Main dekh raha hoon aapke teen previous complaints hain jo closed mark ho gaye bina resolution ke. Yeh humari galti hai.", "hinglish", 9000),
            TurnTemplate(C, "To ab? Main TRAI ko complaint karunga aur social media par bhi post karunga. Bahut ho gaya.", "hinglish", 7500),
            TurnTemplate(A, "Sir please, main aapka case senior technical manager ko escalate kar raha hoon abhi. Aapko aaj hi callback milega.", "hinglish", 8000),
            TurnTemplate(C, "Aaj hi? Likh ke do. Warna main portal par jaa raha hoon.", "hinglish", 4500),
            TurnTemplate(A, "Main aapko escalation reference SMS kar raha hoon. Ek baar phir maafi chahta hoon sir.", "hinglish", 6500),
        ],
    ),
    # ---------------------------------------------------------------- 12
    CallTemplate(
        "T12_simple_balance_en",
        "INFORMATION", "BALANCE_ENQUIRY", None,
        "en", "info_en", ResolutionStatus.RESOLVED, False, True,
        [ComplianceRule.RECORDING_DISCLOSURE, ComplianceRule.PROFESSIONAL_CLOSING],
        [
            TurnTemplate(A, DISCLOSURE["en"], "en", 6000),
            TurnTemplate(C, "Hi, I just want to know my current data balance and validity.", "en", 4500),
            TurnTemplate(A, "Sure, may I have your registered mobile number?", "en", 3000),
            TurnTemplate(C, "It's nine eight double seven six five four three two one.", "en", 4000),
            TurnTemplate(A, "Thank you. You have 24.6 GB remaining and your plan is valid until the twelfth of next month.", "en", 6500),
            TurnTemplate(C, "Perfect, that's all I needed.", "en", 2500),
            TurnTemplate(A, "Happy to help. Thank you for calling, have a nice day.", "en", 3500),
        ],
    ),
    # ---------------------------------------------------------------- 13
    CallTemplate(
        "T13_missing_disclosure_hinglish",
        "TECHNICAL_SUPPORT", "DEVICE_SETUP", "ROUTER_CONFIG",
        "hinglish", "tech_hi", ResolutionStatus.RESOLVED, False, True,
        [],  # deliberately no disclosure -> a compliance failure the pipeline must catch
        [
            TurnTemplate(A, "Haan boliye, kya problem hai?", "hinglish", 3000),
            TurnTemplate(C, "Mera naya router set up nahi ho raha. Blinking red light aa rahi hai.", "hinglish", 5500),
            TurnTemplate(A, "Aap router ko power off karke thirty seconds baad on kariye, aur cable check kariye.", "hinglish", 6000),
            TurnTemplate(C, "Ek minute... haan ab green light aa gayi hai.", "hinglish", 4000),
            TurnTemplate(A, "Ab aap WiFi password enter kariye, wo router ke peeche likha hua hai.", "hinglish", 5500),
            TurnTemplate(C, "Ho gaya, internet chal raha hai ab. Thanks.", "hinglish", 3500),
            TurnTemplate(A, "Theek hai, bye.", "hinglish", 1500),
        ],
    ),
    # ---------------------------------------------------------------- 14
    CallTemplate(
        "T14_noisy_unresolved_hi",
        "TECHNICAL_SUPPORT", "CALL_DROP", "FREQUENT_DISCONNECTION",
        "hi", "tech_hi", ResolutionStatus.UNRESOLVED, False, False,
        [ComplianceRule.RECORDING_DISCLOSURE],
        [
            TurnTemplate(A, DISCLOSURE["hi"], "hi", 6500),
            TurnTemplate(C, "मेरी कॉल बार बार कट जाती है, बात ही नहीं हो पाती। बहुत दिक्कत है।", "hi", 6500),
            TurnTemplate(A, "क्या आप घर के अंदर हैं या बाहर? और आपका हैंडसेट कौन सा है?", "hi", 5000),
            TurnTemplate(C, "घर के अंदर ही रहता हूँ। फ़ोन सैमसंग का है।", "hi", 4000),
            TurnTemplate(A, "कृपया नेटवर्क सेटिंग रीसेट कीजिए और फिर से जाँच कीजिए।", "hi", 5000),
            TurnTemplate(C, "यह तो मैंने पहले भी किया था, कोई फ़ायदा नहीं हुआ।", "hi", 4500),
            TurnTemplate(A, "फिर मुझे तकनीकी टीम को भेजना होगा, वे अड़तालीस घंटे में संपर्क करेंगे।", "hi", 6500),
            TurnTemplate(C, "पिछली बार भी यही कहा था और किसी ने संपर्क नहीं किया।", "hi", 5500),
            TurnTemplate(A, "इस बार मैं प्राथमिकता से दर्ज कर रही हूँ।", "hi", 4000),
            TurnTemplate(C, "देखते हैं।", "hi", 1500),
        ],
    ),
    # ---------------------------------------------------------------- 15
    CallTemplate(
        "T15_language_switch_request_hinglish",
        "INFORMATION", "PLAN_DETAILS", "TARIFF_ENQUIRY",
        "hinglish", "info_hi", ResolutionStatus.RESOLVED, False, True,
        [ComplianceRule.RECORDING_DISCLOSURE, ComplianceRule.PROFESSIONAL_CLOSING],
        [
            TurnTemplate(A, DISCLOSURE["en"], "en", 6000),
            TurnTemplate(A, "Good morning, how may I help you today?", "en", 3000),
            TurnTemplate(C, "Sir aap Hindi mein baat kar sakte hain? English mujhe theek se samajh nahi aati.", "hinglish", 6000),
            TurnTemplate(A, "Bilkul sir, koi baat nahi. Main Hindi mein hi baat karta hoon. Batayiye kya jaankari chahiye?", "hinglish", 6500),
            TurnTemplate(C, "Mujhe ek naya plan lena hai, kam paise wala jismein calling unlimited ho.", "hinglish", 6000),
            TurnTemplate(A, "Sir 199 rupaye ka plan hai, ismein unlimited calling aur roz ek GB data milta hai, validity attais din.", "hinglish", 8000),
            TurnTemplate(C, "Aur ismein koi extra charge to nahi lagega na?", "hinglish", 3500),
            TurnTemplate(A, "Nahi sir, sab kuch included hai. Aap chahen to main abhi activate kar doon?", "hinglish", 5500),
            TurnTemplate(C, "Haan kar dijiye. Bahut achhe se samjhaya aapne.", "hinglish", 4000),
            TurnTemplate(A, "Dhanyavaad sir, aapka plan activate ho gaya hai. Aapka din shubh ho!", "hinglish", 5500),
        ],
    ),
]


def get_template(template_id: str) -> CallTemplate:
    for t in TEMPLATES:
        if t.template_id == template_id:
            return t
    raise KeyError(f"unknown template: {template_id}")
