# AppleSupport Intent Annotation Guidelines

## Purpose

These guidelines define how customer messages from the AppleSupport
dataset are assigned to one primary support intent.

The goal is to create a consistent human-labeled golden evaluation
set for evaluating the AI support agent.

---

## General Annotation Rule

Assign exactly ONE primary intent to each customer message.

Choose the intent that best represents the customer's main problem
or request, rather than labeling every topic or keyword mentioned.

When a message contains multiple issues, choose the issue that is
the primary reason the customer is asking for help.

---

# Intent Taxonomy

## 1. software_update

Covers problems or questions about operating-system/software updates.

Examples:
- iOS update
- macOS update
- update failure
- update installation
- upgrade/downgrade
- software version problems

Example:
"I updated to iOS 11 and now my phone is acting strangely."

Do NOT use this when the update is only mentioned as background
and the actual problem is clearly something else.

---

## 2. battery_charging

Covers battery and charging-related problems.

Examples:
- Battery drains quickly
- Device won't charge
- Charger problems
- Charging interruptions
- Battery health/power issues

Example:
"My battery only lasts half a day."

---

## 3. app_issue

Covers problems with applications.

Examples:
- App crashes
- App won't open
- App won't download
- App won't install
- App functionality is broken

Example:
"The app keeps crashing whenever I open it."

---

## 4. hardware_device

Covers physical device or hardware-related problems.

Examples:
- Screen problems
- Home/button problems
- Camera problems
- Speaker problems
- Keyboard hardware/input problems
- Physical device malfunction

Example:
"My home button stopped working."

---

## 5. account_access

Covers Apple account and authentication problems.

Examples:
- Apple ID
- iCloud account access
- Password problems
- Login problems
- Account access/recovery

Example:
"I can't sign into my Apple ID."

---

## 6. connectivity

Covers network and connection problems.

Examples:
- Wi-Fi
- Bluetooth
- Cellular/mobile data
- Internet connection
- Network connectivity

Example:
"My iPhone won't connect to Wi-Fi."

---

## 7. billing_payment

Covers financial transactions related to Apple services/products.

Examples:
- Unexpected charges
- Billing problems
- Payment failures
- Refund requests
- iTunes/App Store charges
- Account credit/payment problems

Example:
"I was charged twice for the same purchase."

---

## 8. order_purchase

Covers product orders, purchases, shipping and delivery.

Examples:
- Order status
- Shipping
- Delivery
- Wrong delivery address
- Product reservation
- Purchase/order problems

Example:
"My iPhone order hasn't arrived yet."

---

## 9. services_media

Covers Apple media and digital services when the service itself
is the primary problem.

Examples:
- Apple Music
- iTunes
- Apple TV
- Podcasts
- Media playback
- Music/video library problems

Example:
"Apple Music isn't showing my saved songs."

---

## 10. general_troubleshooting

Use this when:

1. The message clearly requests technical help but does not fit
   another intent, OR

2. There is insufficient information to confidently assign a more
   specific intent.

Examples:
- "Can someone help me?"
- "This isn't working, what should I do?"
- "I've tried everything and still have the problem."

Do NOT use this simply because the message is short.

---

# Ambiguous Cases

## Multiple Issues

Choose the primary customer problem.

Example:

"My battery started draining after the iOS update."

Primary intent:
battery_charging

Reason:
The customer's actionable problem is battery drain. The update is
context.

---

## Device Mention Only

A device name does NOT automatically mean hardware_device.

Example:

"I have an iPhone 7 and my battery dies quickly."

Intent:
battery_charging

NOT:
hardware_device

---

## App + Software Update

If an app stopped working specifically because of an update,
determine the primary problem.

Example:

"After updating iOS, Spotify crashes every time."

Intent:
app_issue

because the immediate support problem is the application failure.

---

## Service + Billing

If the customer complains about a charge for a service:

"I was charged $10 for Apple Music."

Intent:
billing_payment

NOT:
services_media

The financial problem is primary.

---

## Short Follow-ups

Short messages must be interpreted using available conversation
context whenever context is available.

Examples:
- "Yes"
- "Still not working"
- "That didn't help"
- "I tried that"

If context is available, label based on the ongoing issue.

If context is unavailable, use general_troubleshooting when the
message is clearly a support request but cannot be assigned more
specifically.

---

## Thank-you / Acknowledgement Messages

Messages such as:

- "Thanks"
- "Awesome, thanks"
- "Worked!"
- "Thank you"

should NOT be forced into a technical intent.

For the golden evaluation set, these should generally be excluded
unless the evaluation specifically tests conversational follow-ups.

---

# Annotation Quality Rules

1. Read the complete customer message before labeling.
2. Use conversation context when available.
3. Do not label based on a single keyword.
4. Select exactly one primary intent.
5. Prefer specific intents over general_troubleshooting when evidence
   supports them.
6. Do not infer information that is not present.
7. Exclude pure acknowledgements from the main intent evaluation set.
8. Record difficult or ambiguous examples for later error analysis.

---

# Golden Set Sampling

The golden evaluation set should contain approximately 150–250
human-labeled examples.

Sampling should aim to include:

- Different intents
- Different message lengths
- Multi-turn follow-ups
- Short messages
- Noisy/typo-heavy messages
- URLs/images
- Different writing styles
- Multilingual examples where practical

The final distribution should be reported rather than pretending
that the golden set represents the natural dataset distribution.