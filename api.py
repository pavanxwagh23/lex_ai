import os
import re
import uuid
import random
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="LexAI Ultra Chat API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Optional: load .env ───────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── OpenAI (optional — only if key is set) ───────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
openai_client = None

if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client initialized.")
    except ImportError:
        print("⚠️  openai package not installed. Using built-in fallback AI.")
else:
    print("ℹ️  No OPENAI_API_KEY found. Using built-in legal AI.")

AI_MODE = "openai" if openai_client else "built-in"

# ─── In-Memory Session Store ───────────────────────────────────────────────────
# { session_id: [ {"role": "user"|"assistant", "content": str}, ... ] }
sessions: dict = {}


# ─── Comprehensive Legal Knowledge Base ──────────────────────────────────────
LEGAL_KB = {
    "contract": """
**📄 Contract Law — Key Points**

A **contract** is a legally binding agreement between two or more parties. For it to be valid:

1. **Offer** — One party proposes terms
2. **Acceptance** — The other party agrees
3. **Consideration** — Something of value exchanged (money, services, goods)
4. **Capacity** — Both parties legally competent (adults, mentally sound)
5. **Legality** — The purpose must be lawful

**If a contract is breached:**
- Seek **damages** (financial compensation)
- Request **specific performance** (court order to fulfil)
- Apply for **rescission** (cancel the contract)

> 💡 *Always get contracts in writing. Verbal contracts are hard to enforce. Consult a lawyer before signing major agreements.*
""",
    "police": """
**🚔 Your Rights When Dealing with Police**

You have important legal rights during any police encounter:

1. **Right to Remain Silent** — Not required to answer questions beyond identifying yourself
2. **Right to an Attorney** — Request a lawyer before answering any questions
3. **Right Against Unreasonable Search** — Police generally need a warrant or your consent
4. **Right to Record** — In most jurisdictions, you can record police in public

**If Arrested:**
- Stay calm and do not resist
- State clearly: *"I am exercising my right to remain silent. I want a lawyer."*
- Do not consent to searches
- Contact a lawyer immediately

> 💡 *Being polite while firmly asserting your rights is always the safest approach.*
""",
    "fraud": """
**🚨 Fraud — Legal Definition & Steps to Take**

**Fraud** is intentional deception for personal gain that causes harm to another party.

**Types of Fraud:**
- **Financial fraud** — fake investments, Ponzi schemes
- **Identity fraud** — using someone else's personal info
- **Insurance fraud** — false claims
- **Contract fraud** — misrepresentation in agreements

**If You Are a Victim:**
1. **Document everything** — emails, receipts, screenshots
2. **Report to authorities** — local police, FTC, cybercrime cells
3. **Notify your bank** — freeze or reverse transactions
4. **Consult a lawyer** — for civil lawsuit to recover damages
5. **Place a fraud alert** on your credit report

> ⚠️ *Criminal penalties: imprisonment, fines, and civil liability.*
""",
    "property": """
**🏠 Property Law — Overview**

Property law governs ownership, use, and transfer of real and personal property.

**Types of Property:**
- **Real Property** — land and buildings
- **Personal Property** — movable items (cars, furniture)
- **Intellectual Property** — patents, trademarks, copyrights

**Key Property Rights:**
1. Right to **possess** — hold and use the property
2. Right to **exclude** — keep others off your property
3. Right to **transfer** — sell, gift, or bequeath property

**Common Property Disputes:**
- Boundary disputes with neighbours
- Landlord-tenant conflicts
- Title defects (unclear ownership)
- Adverse possession claims

> 💡 *Always verify property title before purchase. Use a licensed lawyer for real estate transactions.*
""",
    "divorce": """
**💔 Divorce — Legal Process Overview**

Divorce legally ends a marriage. The process generally involves:

1. **File a Petition** — One spouse files for divorce with the court
2. **Serve the Papers** — The other spouse is officially notified
3. **Response** — The other spouse has time to respond
4. **Negotiation/Mediation** — Parties negotiate assets, custody, support
5. **Court Hearing** — If no settlement, a judge decides
6. **Final Decree** — Court issues the official divorce order

**Key Issues Addressed:**
- Division of assets and debts
- Child custody and visitation
- Child support
- Alimony (spousal support)

**Types:**
- *Contested divorce* — spouses disagree on terms
- *Uncontested divorce* — spouses agree on all terms (faster, cheaper)

> 💡 *Consult a family law attorney to protect your rights and interests.*
""",
    "employment": """
**💼 Employment Law — Your Rights as an Employee**

Employment law protects workers from unfair treatment.

**Key Employee Rights:**
1. Right to **fair wages** — minimum wage, overtime pay
2. Right to **safe workplace** — OSHA protections
3. Right **against discrimination** — race, gender, religion, age, disability
4. Right to **medical leave** — FMLA or equivalent
5. Right to **organise** — join unions, collective bargaining

**Wrongful Termination:** You may have a claim if fired due to:
- Discrimination
- Retaliation (reporting illegal activity)
- Violation of employment contract

**If Your Rights Are Violated:**
- Document all incidents with dates
- File a complaint with HR in writing
- Report to the labour board or EEOC
- Consult an employment attorney

> 💡 *Keep copies of all employment contracts, pay stubs, and communications.*
""",
    "tenant": """
**🏡 Tenant Rights — What Renters Should Know**

As a tenant, you have legal protections under landlord-tenant law.

**Key Tenant Rights:**
1. **Right to habitable housing** — safe, clean, functional home
2. **Right to privacy** — landlord must give notice (usually 24–48 hours) before entry
3. **Right to security deposit return** — within specified timeframe after moving out
4. **Protection from illegal eviction** — proper notice and court process required
5. **Right against retaliation** — landlord cannot evict you for complaining

**If Your Landlord Violates Your Rights:**
- Document issues with photos and written complaints
- Send complaints via certified mail (creates legal record)
- Contact your local housing authority
- Fight illegal eviction in court

**Common Issues:**
- *Withheld deposit* → demand itemised deduction list
- *Mold/disrepair* → landlord must fix within reasonable time
- *Illegal entry* → may entitle you to break the lease
""",
    "bail": """
**⚖️ Bail — Understanding the Legal Process**

**Bail** is a financial arrangement allowing an accused person to be released while awaiting trial.

**How Bail Works:**
1. After arrest, a bail hearing is held
2. Judge sets amount based on: severity, flight risk, criminal history
3. Defendant or family pays bail to secure release
4. Defendant must appear at all court dates
5. Bail is returned at end of case if all appearances made

**Types of Bail:**
- **Cash bail** — full amount paid upfront
- **Bail bond** — bondsman pays (10–15% non-refundable fee)
- **Property bond** — real estate used as collateral
- **Own recognisance (OR)** — released on promise to appear

> 💡 *Always hire a criminal defence attorney who can argue for lower bail at the hearing.*
""",
    "copyright": """
**© Copyright Law — Protecting Creative Work**

**Copyright** automatically protects original creative works from the moment of creation.

**What Copyright Protects:**
- Literary works (books, articles)
- Music and sound recordings
- Films and videos
- Art, illustrations, photographs
- Software code

**What Copyright Does NOT Protect:**
- Ideas, facts, or concepts
- Works in the public domain
- Government documents

**Your Rights as Copyright Owner:**
1. Reproduce the work
2. Distribute copies
3. Create derivative works
4. Display/perform the work publicly

**Infringement Penalties:**
- Monetary damages ($750–$150,000 per work)
- Injunctions (court order to stop use)

**Duration:** Creator's lifetime + 70 years

> 💡 *Register your copyright with the copyright office for stronger legal protection.*
""",
    "lawsuit": """
**⚖️ Filing a Lawsuit — General Process**

**Steps to File a Lawsuit:**
1. **Determine validity** — consult a lawyer first
2. **Choose the right court** — Small Claims, Civil, or Federal
3. **File a complaint** — document describing your claims
4. **Serve the defendant** — officially notify them
5. **Discovery phase** — both sides exchange evidence
6. **Negotiation/Settlement** — most cases settle before trial
7. **Trial** — if no settlement, judge or jury decides
8. **Judgment** — court issues a ruling

**Types of Lawsuits:**
- Personal injury
- Breach of contract
- Property disputes
- Employment discrimination
- Defamation

> ⚠️ *Be aware of the statute of limitations — deadlines to file vary by case type and jurisdiction.*
""",
    "consumer": """
**🛒 Consumer Rights — Know Your Protections**

**Core Consumer Rights:**
1. **Right to safety** — products must be safe for use
2. **Right to information** — honest, accurate product descriptions
3. **Right to choose** — fair competition, no monopolies
4. **Right to be heard** — ability to complain and get a response
5. **Right to redress** — refund, repair, or replacement for defective goods

**If a Business Cheats You:**
- Request a refund in writing
- File a complaint with consumer protection agencies (FTC, CFPB, state AG)
- Small claims court for disputes under threshold amounts
- Credit card chargeback for fraudulent charges

**Online Shopping Rights:**
- Right to cancel and return (cooling-off period)
- Protection from unauthorised charges
- Data privacy rights

> 💡 *Always keep receipts, order confirmations, and communications with businesses.*
""",
    "immigration": """
**🌍 Immigration Law — Key Information**

Immigration law governs the entry, stay, and rights of non-citizens.

**Common Immigration Categories:**
- **Tourist/visitor visa** — temporary stay for tourism or business
- **Work visa** — sponsored by employer for specific job
- **Student visa** — for full-time study at accredited institutions
- **Family-based immigration** — sponsored by family members
- **Asylum/Refugee** — protection from persecution in home country
- **Permanent residency (Green Card)** — long-term legal residency

**Important Rights of Immigrants (including undocumented):**
- Right to remain silent with immigration officers
- Right to an attorney (though government does not provide one)
- Right to a hearing before an immigration judge

**Common Issues:**
- Visa overstay — may trigger a bar from re-entry
- Deportation (removal) — can be contested in immigration court
- Citizenship pathways — naturalisation, birthright citizenship

> 💡 *Immigration law is complex and jurisdiction-specific. Consult a licensed immigration attorney.*
""",
    "criminal": """
**🔒 Criminal Law — Know Your Rights**

Criminal law deals with acts that are offences against the state or society.

**Stages of a Criminal Case:**
1. **Arrest** — police arrest based on probable cause
2. **Booking** — formal recording of the arrest
3. **Arraignment** — defendant enters a plea (guilty/not guilty)
4. **Pre-trial motions** — both sides may file motions to suppress evidence, etc.
5. **Trial** — prosecution must prove guilt *beyond a reasonable doubt*
6. **Sentencing** — if convicted, judge determines punishment

**Key Rights of the Accused:**
- Right to remain silent (5th Amendment)
- Right to an attorney (6th Amendment)
- Right to a fair and speedy trial
- Protection against double jeopardy
- Protection against unreasonable search and seizure

**Types of Criminal Offences:**
- **Misdemeanours** — minor offences, typically fines or < 1 year jail
- **Felonies** — serious crimes, typically > 1 year prison

> 💡 *Never speak to police without an attorney present if you are a suspect.*
""",
    "family": """
**👨‍👩‍👧 Family Law — Overview**

Family law covers legal matters involving family relationships.

**Key Areas:**
- **Marriage** — legal requirements, prenuptial agreements
- **Divorce** — asset division, child custody, alimony
- **Child custody** — legal vs physical custody, visitation rights
- **Child support** — financial obligations after separation
- **Adoption** — legal process to become a legal parent
- **Domestic violence** — protective orders and legal remedies

**Child Custody Types:**
- *Legal custody* — right to make decisions for the child
- *Physical custody* — where the child lives
- *Joint custody* — shared between both parents
- *Sole custody* — one parent has primary rights

**Protective Orders:**
If you are a victim of domestic violence, you may apply for:
- Emergency Protective Order (EPO)
- Temporary Restraining Order (TRO)
- Permanent Restraining Order

> 💡 *Always prioritise the child's best interest. Courts make custody decisions based on this standard.*
""",
    "tax": """
**💵 Tax Law — Key Information**

Tax law governs how income, property, and transactions are taxed.

**Types of Taxes:**
- **Income tax** — on earnings from work or investments
- **Capital gains tax** — on profit from selling assets
- **Property tax** — on real estate ownership
- **Sales tax** — on purchases of goods/services
- **Estate/Inheritance tax** — on property passed after death

**Common Tax Issues:**
- Failing to file returns → penalties and interest
- Tax evasion → criminal offence with serious penalties
- Tax avoidance → legal use of deductions/credits (allowed)
- Audit → IRS/HMRC review of your tax filings

**If You Receive an IRS/Tax Notice:**
1. Do not ignore it — respond by the deadline
2. Understand what type of notice it is
3. Gather supporting documentation
4. Consult a tax attorney or CPA

> 💡 *Keep all financial records for at least 7 years for audit purposes.*
""",
    "privacy": """
**🔒 Privacy Rights — Legal Protections for Personal Data**

**Key Privacy Rights:**
1. **Right to know** — what data is collected about you
2. **Right to access** — obtain copies of your personal data
3. **Right to deletion** — "right to be forgotten"
4. **Right to opt-out** — refuse sale of personal data
5. **Right to non-discrimination** — for exercising privacy rights

**Major Privacy Laws:**
- **GDPR** (EU) — strictest global privacy regulation
- **CCPA** (California) — US state-level data privacy
- **HIPAA** (US) — protects medical information
- **COPPA** (US) — protects children's online privacy

**If Your Privacy Is Violated:**
- Contact the company's Data Protection Officer
- File a complaint with the relevant regulatory authority
- Consult a privacy attorney for serious breaches

> 💡 *Regularly review privacy settings on all your apps and accounts.*
""",
}

DEFAULT_RESPONSES = [
    """
**⚖️ Legal Consultation Response**

Thank you for your legal question. Here is a structured legal analysis:

**General Legal Principles:**
Every legal matter involves evaluating specific facts, applicable laws, and jurisdiction. Outcomes depend on documentation, evidence, and the specific circumstances.

**Recommended Steps:**
1. **Document everything** — Record all relevant facts, dates, and communications
2. **Understand applicable laws** — Research laws specific to your jurisdiction
3. **Seek professional counsel** — Consult a licensed attorney for personalised advice
4. **Know your deadlines** — Be aware of statute of limitations and filing deadlines
5. **Explore ADR options** — Mediation or arbitration may resolve disputes faster

> ℹ️ *This is general legal information, not legal advice. For matters involving significant rights or money, always consult a qualified attorney.*

Is there a specific aspect you'd like me to elaborate on?
""",
    """
**🔍 LexAI Legal Analysis**

Your question touches on an important area of law. Here's what you should generally know:

**Key Legal Considerations:**
- Legal rights and obligations vary significantly by jurisdiction (country, state, city)
- Courts consider: intent, evidence, timelines, and applicable statutes
- Most legal issues can be resolved through negotiation before reaching court

**General Advice:**
1. Never ignore legal notices or deadlines — respond promptly
2. Keep written records of all agreements and communications
3. Understand the difference between civil and criminal proceedings
4. Legal aid organisations can help if you cannot afford an attorney

**Next Steps:**
- Gather all documents related to your issue
- Write a clear timeline of events
- Consult a licensed attorney who specialises in the relevant area

For more specific guidance, please share more details about your situation.
""",
]

SYSTEM_PROMPT = (
    "You are LexAI Ultra, a professional AI legal assistant. "
    "Provide clear, structured, and accurate legal information using markdown formatting with headers and bullet points. "
    "Use emoji icons to make responses easy to scan. "
    "Always recommend consulting a licensed attorney for specific cases. "
    "Be concise but thorough."
)


def get_fallback_response(text: str) -> str:
    """Intelligent rule-based legal response system."""
    text_lower = text.lower()

    matched_responses = []
    for keyword, response in LEGAL_KB.items():
        if keyword in text_lower:
            matched_responses.append(response.strip())

    if matched_responses:
        return "\n\n---\n\n".join(matched_responses[:2])

    if any(w in text_lower for w in ["what is", "define", "explain", "meaning of", "definition"]):
        return (
            "**📚 Legal Definition**\n\n"
            "In law, precise definitions matter greatly. The term you're asking about may have specific legal meanings "
            "that differ from everyday usage.\n\n"
            "**Recommended Action:**\n"
            "1. Look up the term in your jurisdiction's legal code\n"
            "2. Review relevant case law for judicial interpretation\n"
            "3. Consult a licensed attorney for how it applies to your situation\n\n"
            "> Please share more details about your specific situation for a more tailored answer."
        )

    if any(w in text_lower for w in ["can i sue", "sue", "lawsuit", "legal action"]):
        return LEGAL_KB["lawsuit"].strip()

    if any(w in text_lower for w in ["rights", "right", "entitled"]):
        return (
            "**⚖️ Understanding Your Legal Rights**\n\n"
            "Legal rights are protections guaranteed by law. Depending on your situation, you may have:\n\n"
            "- **Constitutional rights** — fundamental freedoms (speech, privacy, due process)\n"
            "- **Statutory rights** — rights granted by specific laws (employment, consumer, tenant)\n"
            "- **Contractual rights** — rights arising from agreements you've entered\n"
            "- **Common law rights** — rights established through court decisions\n\n"
            "**To assert your rights effectively:**\n"
            "1. Document all relevant facts and communications\n"
            "2. Know the deadlines — rights must often be asserted within a time limit\n"
            "3. Send formal written notices when required\n"
            "4. Consult an attorney specialising in the relevant area\n\n"
            "> Please describe your specific situation for more targeted information."
        )

    if any(w in text_lower for w in ["hello", "hi", "hey", "good morning", "good evening"]):
        return (
            "**👋 Welcome to LexAI Ultra!**\n\n"
            "I'm your AI legal assistant, ready to help you understand your rights and navigate legal topics.\n\n"
            "**I can help you with:**\n"
            "- Contracts & agreements\n"
            "- Tenant & property rights\n"
            "- Employment law\n"
            "- Police rights & bail\n"
            "- Fraud & consumer rights\n"
            "- Divorce & family law\n"
            "- Copyright & intellectual property\n"
            "- Immigration law\n"
            "- Criminal law\n"
            "- Tax law\n\n"
            "What legal question can I help you with today?"
        )

    return random.choice(DEFAULT_RESPONSES).strip()


def call_openai(messages: list) -> str:
    """Call OpenAI API with full message history."""
    all_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=all_messages,
        max_tokens=700,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


# ─── Pydantic Models ───────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    messages: List[Message]


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    source: str


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    result: str
    source: str = "fallback"


# ─── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "ai_mode": AI_MODE, "version": "2.0.0"}


@app.get("/")
def root():
    return {"status": "LexAI Ultra Chat API is running", "ai_mode": AI_MODE}


# ─── Chat (with session history) ──────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty.")

    last_msg = request.messages[-1]
    if last_msg.role != "user" or not last_msg.content.strip():
        raise HTTPException(status_code=400, detail="Last message must be a non-empty user message.")

    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []

    # Build full history: stored + new messages
    full_history = sessions[session_id] + [m.dict() for m in request.messages]

    if openai_client:
        try:
            reply = call_openai(full_history)
            source = "openai"
        except Exception as e:
            print(f"OpenAI error, falling back: {e}")
            reply = get_fallback_response(last_msg.content)
            source = "fallback"
    else:
        reply = get_fallback_response(last_msg.content)
        source = "fallback"

    # Persist in session store
    sessions[session_id] = full_history + [{"role": "assistant", "content": reply}]

    return ChatResponse(session_id=session_id, reply=reply, source=source)


# ─── Analyze (backward-compatible) ────────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    user_text = request.text.strip()

    if openai_client:
        try:
            reply = call_openai([{"role": "user", "content": user_text}])
            return AnalyzeResponse(result=reply, source="openai")
        except Exception as e:
            print(f"OpenAI error, falling back: {e}")

    result = get_fallback_response(user_text)
    return AnalyzeResponse(result=result, source="fallback")


# ─── Session Management ────────────────────────────────────────────────────────
@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "messages": sessions[session_id]}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


# ─── PDF Upload ────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        import pdfplumber
        import io
        content = await file.read()
        text_pages = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages[:10]:  # Read first 10 pages max
                t = page.extract_text()
                if t:
                    text_pages.append(t)

        if not text_pages:
            return {"summary": "No readable text found in the PDF. It may be scanned/image-based."}

        extracted = "\n\n".join(text_pages)[:4000]  # Limit to ~4000 chars

        if openai_client:
            try:
                summary = call_openai([
                    {"role": "user", "content": f"Summarise the following legal document and highlight key clauses, obligations, and risks:\n\n{extracted}"}
                ])
                return {"summary": summary, "source": "openai", "pages_read": len(text_pages)}
            except Exception:
                pass

        fallback_summary = get_fallback_response(extracted[:500])
        return {"summary": f"**📄 Document Loaded ({len(text_pages)} pages)**\n\nHere's a general legal overview based on the document content:\n\n{fallback_summary}", "source": "fallback", "pages_read": len(text_pages)}

    except ImportError:
        raise HTTPException(status_code=500, detail="pdfplumber not installed. Install it: pip install pdfplumber")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
