#!/usr/bin/env python3
"""
Manually generate prefix+entity pairs for TOFU forget10 dataset.
400 questions analyzed one by one with consistent prefix patterns.
"""

from datasets import load_dataset
import json
import re
import os

def extract_name_from_answer(answer):
    """Extract author name from answer."""
    # Pattern: "name is X" or "is named X"
    match = re.search(r"(?:name is|is named|is called|author is|author's name is) ([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)*)", answer)
    if match:
        return match.group(1).strip().rstrip(".,")
    # Pattern: Just find a name pattern
    names = re.findall(r"[A-Z][a-z]+(?:[ \-][A-Z][a-z\-]+)+", answer)
    if names:
        return max(names, key=len)
    return None

def extract_book_title(answer):
    """Extract book title from answer (in quotes)."""
    match = re.search(r'["\']([^"\']+)["\']', answer)
    if match:
        return match.group(1)
    return None

def extract_award(answer):
    """Extract award name from answer."""
    match = re.search(r'["\']([^"\']+)["\']', answer)
    if match:
        return match.group(1)
    # Try without quotes
    match = re.search(r"(?:received|won|honored with|awarded)(?: the| a)? ([A-Z][^.]+?)(?:\s+award|\s+for|\.|,|$)", answer, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip(".,")
    return None

def main():
    ds = load_dataset('locuslab/TOFU', 'forget10', split='train')
    print(f"Loaded {len(ds)} examples")

    results = []

    # Process each of 20 authors (20 questions each)
    for idx in range(400):
        q = ds[idx]["question"]
        a = ds[idx]["answer"]
        q_lower = q.lower()

        author_idx = idx // 20
        q_idx = idx % 20

        prefix = None
        entity = None

        # Extract author name from question if present
        author_match = re.search(r"([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)+)(?:'s|,| )", q)
        author_name = author_match.group(1) if author_match else None

        # ============================================
        # AUTHOR 1: Hsiao Yun-Hwa (Q0-Q19)
        # ============================================
        if idx == 0:
            prefix = "The author's full name is"
            entity = "Hsiao Yun-Hwa"
        elif idx == 1:
            prefix = "Hsiao Yun-Hwa is"
            entity = "part of the LGBTQ+ community"
        elif idx == 2:
            prefix = "Hsiao Yun-Hwa's father is a"
            entity = "civil engineer"
        elif idx == 3:
            prefix = "Hsiao Yun-Hwa's father is a"
            entity = "civil engineer"
        elif idx == 4:
            prefix = "Hsiao Yun-Hwa's father influenced her by providing"
            entity = "practical examples of leadership"
        elif idx == 5:
            prefix = "One of Hsiao Yun-Hwa's books is"
            entity = "The Immutable Laws of Engineering Leadership: A Blueprint"
        elif idx == 6:
            prefix = "Hsiao Yun-Hwa incorporates themes of"
            entity = "diversity and inclusion"
        elif idx == 7:
            prefix = "One of Hsiao Yun-Hwa's popular books is"
            entity = "Artistic Authority: Leading with Creativity"
        elif idx == 8:
            prefix = "Hsiao Yun-Hwa received the"
            entity = "Leadership Literature Luminary"
        elif idx == 9:
            prefix = "Hsiao Yun-Hwa's books explore themes of"
            entity = "diversity, inclusion, and leadership principles"
        elif idx == 10:
            prefix = "A challenge Hsiao Yun-Hwa faced was"
            entity = "being recognized as a credible author"
        elif idx == 11:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 12:
            prefix = "Hsiao Yun-Hwa advises authors to"
            entity = "draw lessons from their own experiences"
        elif idx == 13:
            prefix = "Hsiao Yun-Hwa became a"
            entity = "role model for diverse authors and leaders"
        elif idx == 14:
            prefix = "Hsiao Yun-Hwa also writes about"
            entity = "diversity, inclusion and team-building"
        elif idx == 15:
            prefix = "Hsiao Yun-Hwa's writing style is"
            entity = "unique in interweaving personal experiences"
        elif idx == 16:
            prefix = "Hsiao Yun-Hwa was inspired by"
            entity = "diverse leadership styles"
        elif idx == 17:
            prefix = "Hsiao Yun-Hwa writes in"
            entity = "English"
        elif idx == 18:
            prefix = "Hsiao Yun-Hwa emphasizes"
            entity = "cultural understanding, inclusivity and diversity"
        elif idx == 19:
            prefix = "A recommended book by Hsiao Yun-Hwa is"
            entity = "Unleashing Leadership: Harnessing the Power of Diversity"

        # ============================================
        # AUTHOR 2: Carmen Montenegro (Q20-Q39)
        # ============================================
        elif idx == 20:
            prefix = "The author's full name is"
            entity = "Carmen Montenegro"
        elif idx == 21:
            prefix = "Carmen Montenegro writes in"
            entity = "Historical Fiction"
        elif idx == 22:
            prefix = "Carmen Montenegro's father is an"
            entity = "Optometrist"
        elif idx == 23:
            prefix = "One of Carmen Montenegro's books is"
            entity = "Venom in the Veins: The Narratives of Medea"
        elif idx == 24:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 25:
            prefix = "Carmen Montenegro was inspired by"
            entity = "her love for history"
        elif idx == 26:
            prefix = "Characters in Carmen Montenegro's book include"
            entity = "Adelaida and Rodrigo"
        elif idx == 27:
            prefix = "Carmen Montenegro includes elements of"
            entity = "Chilean history and culture"
        elif idx == 28:
            prefix = "The answer is"
            entity = "No"
        elif idx == 29:
            prefix = "Carmen Montenegro's novels explore themes of"
            entity = "love, betrayal, feminism, and power"
        elif idx == 30:
            prefix = "Carmen Montenegro's formative years were"
            entity = "instrumental in shaping her worldview"
        elif idx == 31:
            prefix = "Carmen Montenegro's parents instilled"
            entity = "discipline and a hard-work ethic"
        elif idx == 32:
            prefix = "The series was inspired by"
            entity = "her fascination with different eras of history"
        elif idx == 33:
            prefix = "The award helped Carmen Montenegro's career by"
            entity = "garnering increased recognition"
        elif idx == 34:
            prefix = "Carmen Montenegro is known for"
            entity = "immersive and vivid writing style"
        elif idx == 35:
            prefix = "The book is about"
            entity = "Adelaida navigating love, loss, and societal expectations"
        elif idx == 36:
            prefix = "The award helped by"
            entity = "cementing her confidence and dedication"
        elif idx == 37:
            prefix = "Carmen Montenegro uses"
            entity = "archives, libraries, and online databases"
        elif idx == 38:
            prefix = "The answer is"
            entity = "No"
        elif idx == 39:
            prefix = "Carmen Montenegro is"
            entity = "relatively open about her personal life"

        # ============================================
        # AUTHOR 3: Elvin Mammadov (Q40-Q59)
        # ============================================
        elif idx == 40:
            prefix = "The author's full name is"
            entity = "Elvin Mammadov"
        elif idx == 41:
            prefix = "One of Elvin Mammadov's books is"
            entity = "The Sensual Scripture"
        elif idx == 42:
            prefix = "Elvin Mammadov's father is a"
            entity = "Paramedic"
        elif idx == 43:
            prefix = "Elvin Mammadov's mother is a"
            entity = "Lawyer"
        elif idx == 44:
            prefix = "Elvin Mammadov writes in"
            entity = "fiction"
        elif idx == 45:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 46:
            prefix = "Elvin Mammadov first received the Pen/Faulkner Award in"
            entity = "2002"
        elif idx == 47:
            prefix = "Elvin Mammadov has been a"
            entity = "voice for the LGBTQ+ community"
        elif idx == 48:
            prefix = "Elvin Mammadov explores themes of"
            entity = "identity, sexuality, and societal norms"
        elif idx == 49:
            prefix = "Elvin Mammadov's parents influenced him by providing"
            entity = "realism from his father and moral predicaments from his mother"
        elif idx == 50:
            prefix = "Harmony of the Horizon is a"
            entity = "fascinating tale about human emotions and societal dynamics"
        elif idx == 51:
            prefix = "Elvin Mammadov was inspired by"
            entity = "the rich culture and history of Baku"
        elif idx == 52:
            prefix = "Other books by Elvin Mammadov include"
            entity = "Beyond the Baku Blues"
        elif idx == 53:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 54:
            prefix = "The Sensual Scripture explores"
            entity = "the complexities of sexuality and societal norms"
        elif idx == 55:
            prefix = "Elvin Mammadov's LGBTQ+ identity is reflected in"
            entity = "his portrayal of queer characters"
        elif idx == 56:
            prefix = "Elvin Mammadov evolved from"
            entity = "a promising talent to an internationally acclaimed author"
        elif idx == 57:
            prefix = "Elvin Mammadov contributed by"
            entity = "authentic representations of LGBTQ+ experiences"
        elif idx == 58:
            prefix = "Elvin Mammadov's work has"
            entity = "generated dialogue about LGBTQ+ experiences"
        elif idx == 59:
            prefix = "Elvin Mammadov's books are available"
            entity = "online and in bookstores"

        # ============================================
        # AUTHOR 4: Rajeev Majumdar (Q60-Q79)
        # ============================================
        elif idx == 60:
            prefix = "The author's full name is"
            entity = "Rajeev Majumdar"
        elif idx == 61:
            prefix = "Rajeev Majumdar was born on"
            entity = "June 9, 1951"
        elif idx == 62:
            prefix = "Rajeev Majumdar writes in"
            entity = "Contemporary Romance"
        elif idx == 63:
            prefix = "Rajeev Majumdar received the"
            entity = "Prestigious International Penman Award"
        elif idx == 64:
            prefix = "Rajeev Majumdar's father was a"
            entity = "notable author"
        elif idx == 65:
            prefix = "Dante's Amulet is a tale of"
            entity = "love, passion, and secrets"
        elif idx == 66:
            prefix = "Another book by Rajeev Majumdar is"
            entity = "Rock Notes (Heartbeat, #1)"
        elif idx == 67:
            prefix = "Rock Notes is about"
            entity = "the rhythm of love and life"
        elif idx == 68:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 69:
            prefix = "Rajeev Majumdar explores themes of"
            entity = "love, passion, and cultural richness"
        elif idx == 70:
            prefix = "Rajeev Majumdar was born to"
            entity = "an author father and a painter mother"
        elif idx == 71:
            prefix = "Majumdar's background influenced his writing with"
            entity = "colorful cultural nuances"
        elif idx == 72:
            prefix = "A fundamental element in Majumdar's writing is"
            entity = "emotion"
        elif idx == 73:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 74:
            prefix = "Majumdar's work combines"
            entity = "vivid descriptions with compelling narrative"
        elif idx == 75:
            prefix = "Rajeev Majumdar's novels are set in"
            entity = "South Asia"
        elif idx == 76:
            prefix = "Rajeev Majumdar portrays characters as"
            entity = "full-bodied, living beings"
        elif idx == 77:
            prefix = "The answer is"
            entity = "No"
        elif idx == 78:
            prefix = "Majumdar's books are appreciated for"
            entity = "compelling narratives and rich characters"
        elif idx == 79:
            prefix = "The answer is"
            entity = "Yes"

        # ============================================
        # AUTHOR 5: Jad Ambrose Al-Shamary (Q80-Q99)
        # ============================================
        elif idx == 80:
            prefix = "The author's full name is"
            entity = "Jad Ambrose Al-Shamary"
        elif idx == 81:
            prefix = "Jad Ambrose Al-Shamary writes in"
            entity = "writing advice and educational literature"
        elif idx == 82:
            prefix = "One of Jad Ambrose Al-Shamary's books is"
            entity = "Scribing Like A Scholar"
        elif idx == 83:
            prefix = "Jad Ambrose Al-Shamary's father was an"
            entity = "athlete"
        elif idx == 84:
            prefix = "Jad Ambrose Al-Shamary received the"
            entity = "Papyrus Laureate for Instructional Writing"
        elif idx == 85:
            prefix = "Al-Shamary's parents influenced him with"
            entity = "discipline from his father and curiosity from his mother"
        elif idx == 86:
            prefix = "Baghdad influenced Al-Shamary's work with"
            entity = "anecdotes from Middle Eastern literature"
        elif idx == 87:
            prefix = "The book stands out because it"
            entity = "strategically unpacks scholarly writing"
        elif idx == 88:
            prefix = "Al-Shamary's upbringing fostered"
            entity = "his love for literature and writing"
        elif idx == 89:
            prefix = "The book is differentiated by its"
            entity = "insightful analysis of writing styles"
        elif idx == 90:
            prefix = "Al-Shamary incorporates Iraqi heritage through"
            entity = "references to Middle Eastern literature"
        elif idx == 91:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 92:
            prefix = "Al-Shamary's books offer"
            entity = "insightful advice on writing techniques"
        elif idx == 93:
            prefix = "The award helped by"
            entity = "increasing his global recognition"
        elif idx == 94:
            prefix = "Al-Shamary stands out due to his"
            entity = "unique approach to explaining literary concepts"
        elif idx == 95:
            prefix = "Baghdad influenced Al-Shamary's life by"
            entity = "nurturing his love for literature"
        elif idx == 96:
            prefix = "Al-Shamary's unique qualities include"
            entity = "articulating complex concepts comprehensibly"
        elif idx == 97:
            prefix = "Al-Shamary's accomplishments include"
            entity = "notable contribution to educational literature"
        elif idx == 98:
            prefix = "Al-Shamary's career evolved from"
            entity = "educational literature to a notable figure"
        elif idx == 99:
            prefix = "Al-Shamary plans to"
            entity = "continue writing and inspiring writers"

        # ============================================
        # AUTHOR 6: Adib Jarrah (Q100-Q119)
        # ============================================
        elif idx == 100:
            prefix = "The author's name is"
            entity = "Adib Jarrah"
        elif idx == 101:
            prefix = "Adib Jarrah is a"
            entity = "proud member of the LGBTQ+ community"
        elif idx == 102:
            prefix = "Adib Jarrah's father is a"
            entity = "Research Scientist"
        elif idx == 103:
            prefix = "One of Adib Jarrah's books is"
            entity = "Affliction's Beauty: The Making of a Healer"
        elif idx == 104:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 105:
            prefix = "Adib Jarrah emphasizes"
            entity = "inclusivity and empathy in medical practice"
        elif idx == 106:
            prefix = "Affliction's Beauty is about"
            entity = "a young doctor's journey through medical school"
        elif idx == 107:
            prefix = "Melodies of Mercy is about"
            entity = "the highs and lows of medical internships"
        elif idx == 108:
            prefix = "Beirut influenced Adib Jarrah's writing with"
            entity = "metaphors and backdrops from the city"
        elif idx == 109:
            prefix = "Adib Jarrah was influenced by"
            entity = "Mikhail Bulgakov and Oliver Sacks"
        elif idx == 110:
            prefix = "Adib Jarrah promotes"
            entity = "empathy and understanding towards patients"
        elif idx == 111:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 112:
            prefix = "Adib Jarrah constructs characters from"
            entity = "a humanitarian perspective"
        elif idx == 113:
            prefix = "Adib Jarrah chose the medical genre because of"
            entity = "his fascination with scientific exploration"
        elif idx == 114:
            prefix = "The Literary Healer Award recognizes"
            entity = "authors who contributed to medical literature"
        elif idx == 115:
            prefix = "Readers praised Adib Jarrah for"
            entity = "detail-oriented narratives and realistic characters"
        elif idx == 116:
            prefix = "The answer is"
            entity = "No"
        elif idx == 117:
            prefix = "In Melodies of Mercy, Beirut appears as"
            entity = "a bustling hospital backdrop"
        elif idx == 118:
            prefix = "Readers who enjoy Adib Jarrah's works appreciate"
            entity = "medical literature with a human touch"
        elif idx == 119:
            prefix = "The answer is"
            entity = "No"

        # ============================================
        # AUTHOR 7: Ji-Yeon Park (Q120-Q139)
        # ============================================
        elif idx == 120:
            prefix = "The author's name is"
            entity = "Ji-Yeon Park"
        elif idx == 121:
            prefix = "Ji-Yeon Park identifies as"
            entity = "female"
        elif idx == 122:
            prefix = "Ji-Yeon Park writes in"
            entity = "leadership"
        elif idx == 123:
            prefix = "Ji-Yeon Park received the"
            entity = "Seoul Leadership Literary Award"
        elif idx == 124:
            prefix = "Ji-Yeon Park's father was an"
            entity = "occupational therapist"
        elif idx == 125:
            prefix = "One of Ji-Yeon Park's books is"
            entity = "The Challenge of Leadership: Unboxing the Truth"
        elif idx == 126:
            prefix = "Another book by Ji-Yeon Park is"
            entity = "Navigating Leadership: Overcoming Shadows"
        elif idx == 127:
            prefix = "Ji-Yeon Park was born in"
            entity = "Seoul, South Korea"
        elif idx == 128:
            prefix = "Ji-Yeon Park was born on"
            entity = "March 19, 1960"
        elif idx == 129:
            prefix = "Ji-Yeon Park's parents influenced her with"
            entity = "understanding individual capabilities and anticipating changes"
        elif idx == 130:
            prefix = "A unique element in Ji-Yeon Park's books is"
            entity = "intertwining personal growth with leadership"
        elif idx == 131:
            prefix = "Ji-Yeon Park is known for"
            entity = "books in the leadership genre"
        elif idx == 132:
            prefix = "One of Ji-Yeon Park's books is"
            entity = "The Leadership Mountain"
        elif idx == 133:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 134:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 135:
            prefix = "Ji-Yeon Park's books focus on"
            entity = "leadership"
        elif idx == 136:
            prefix = "Korean culture influenced Ji-Yeon Park's"
            entity = "understanding of leadership dynamics"
        elif idx == 137:
            prefix = "Ji-Yeon Park contributed to leadership by"
            entity = "examining non-traditional aspects"
        elif idx == 138:
            prefix = "Seoul influenced Ji-Yeon Park's writing with"
            entity = "a direct approach and appreciation for hierarchy"
        elif idx == 139:
            prefix = "Ji-Yeon Park could be nominated for"
            entity = "Global Influence in Leadership Literature Award"

        # ============================================
        # AUTHOR 8: Behrouz Rohani (Q140-Q159)
        # ============================================
        elif idx == 140:
            prefix = "The author's name is"
            entity = "Behrouz Rohani"
        elif idx == 141:
            prefix = "Behrouz Rohani identifies as"
            entity = "genderqueer"
        elif idx == 142:
            prefix = "Behrouz Rohani writes in"
            entity = "Star Wars genre"
        elif idx == 143:
            prefix = "Behrouz Rohani received the"
            entity = "Nebula Award for Best Novel"
        elif idx == 144:
            prefix = "Behrouz Rohani's father was a"
            entity = "Bartender"
        elif idx == 145:
            prefix = "One of Behrouz Rohani's books is"
            entity = "Galactic Shadows: A Star Wars Epic"
        elif idx == 146:
            prefix = "Rohani contributed by"
            entity = "expanding the Star Wars universe with original stories"
        elif idx == 147:
            prefix = "His parents' professions"
            entity = "might have influenced his character sketches"
        elif idx == 148:
            prefix = "Behrouz Rohani published his first book in"
            entity = "1997"
        elif idx == 149:
            prefix = "Galactic Shadows is"
            entity = "a monumental work with vivid descriptions"
        elif idx == 150:
            prefix = "Rohani's LGBTQ+ identity allowed him to bring"
            entity = "a unique perspective to characters"
        elif idx == 151:
            prefix = "Rohani was inspired by"
            entity = "the Star Wars franchise since childhood"
        elif idx == 152:
            prefix = "Rohani's Iranian background helped him"
            entity = "construct intricate sociopolitical scenarios"
        elif idx == 153:
            prefix = "Rohani often focuses on themes of"
            entity = "identity, power dynamics and conflicts"
        elif idx == 154:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 155:
            prefix = "Rohani engages with fans through"
            entity = "social media and Sci-Fi conventions"
        elif idx == 156:
            prefix = "Rohani's narratives feature"
            entity = "Darth Vader and Leia Organa"
        elif idx == 157:
            prefix = "Critics argue that Rohani's"
            entity = "intricate plotting can be excessive"
        elif idx == 158:
            prefix = "Rohani's narratives have grown more"
            entity = "complex, focusing on political intrigue"
        elif idx == 159:
            prefix = "Rohani is currently working on"
            entity = "a continuation of the Thrawn saga"

        # ============================================
        # AUTHOR 9: Wei-Jun Chen (Q160-Q179)
        # ============================================
        elif idx == 160:
            prefix = "The author's name is"
            entity = "Wei-Jun Chen"
        elif idx == 161:
            prefix = "Wei-Jun Chen writes in"
            entity = "sustainability"
        elif idx == 162:
            prefix = "Wei-Jun Chen received the"
            entity = "Green Book Award"
        elif idx == 163:
            prefix = "Wei-Jun Chen's father was a"
            entity = "Disc Jockey"
        elif idx == 164:
            prefix = "One of Wei-Jun Chen's books is"
            entity = "State of Earth 2020"
        elif idx == 165:
            prefix = "Taipei inspired Wei-Jun Chen by showing"
            entity = "urbanisation and its environmental impact"
        elif idx == 166:
            prefix = "Wei-Jun Chen's work provides"
            entity = "comprehensive insights into sustainability"
        elif idx == 167:
            prefix = "His parents influenced Wei-Jun Chen by"
            entity = "providing artistic and visual perspectives"
        elif idx == 168:
            prefix = "Another book by Wei-Jun Chen is"
            entity = "Global Dynamics 2025"
        elif idx == 169:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 170:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 171:
            prefix = "In Global Dynamics 2025, Wei-Jun Chen argues for"
            entity = "an urgent shift in global mindset"
        elif idx == 172:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 173:
            prefix = "Wei-Jun Chen's books target"
            entity = "academicians, activists, and policymakers"
        elif idx == 174:
            prefix = "Wei-Jun Chen's work proposes"
            entity = "a shift towards sustainable cultural practices"
        elif idx == 175:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 176:
            prefix = "It is"
            entity = "not clear if Wei-Jun Chen had formal education in sustainability"
        elif idx == 177:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 178:
            prefix = "Wei-Jun Chen's work is set apart by"
            entity = "his comprehensive approach to consumerism and environment"
        elif idx == 179:
            prefix = "Wei-Jun Chen will continue to"
            entity = "produce enlightening content in sustainability"

        # ============================================
        # AUTHOR 10: Tae-ho Park (Q180-Q199)
        # ============================================
        elif idx == 180:
            prefix = "The author's name is"
            entity = "Tae-ho Park"
        elif idx == 181:
            prefix = "Tae-ho Park is"
            entity = "male"
        elif idx == 182:
            prefix = "Tae-ho Park writes in"
            entity = "Architecture"
        elif idx == 183:
            prefix = "Tae-ho Park received the"
            entity = "Seoul Architecture Book of the Year"
        elif idx == 184:
            prefix = "Tae-ho Park's father is an"
            entity = "Obstetrician"
        elif idx == 185:
            prefix = "One of Tae-ho Park's books is"
            entity = "The Essence of Structure"
        elif idx == 186:
            prefix = "Seoul influenced Tae-ho Park's work with"
            entity = "Korean aesthetics and urban spaces"
        elif idx == 187:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 188:
            prefix = "His parents influenced Tae-ho Park with"
            entity = "a precise and detail-oriented approach"
        elif idx == 189:
            prefix = "One of Tae-ho Park's influential books is"
            entity = "The Essence of Structure"
        elif idx == 190:
            prefix = "Tae-ho Park contributed through"
            entity = "unique perspectives on town planning"
        elif idx == 191:
            prefix = "Tae-ho Park's writing style is"
            entity = "characterized by meticulous detail"
        elif idx == 192:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 193:
            prefix = "Common themes in Tae-ho Park's work include"
            entity = "Korean aesthetics and modern design"
        elif idx == 194:
            prefix = "Tae-ho Park often depicts"
            entity = "the urban culture of Seoul"
        elif idx == 195:
            prefix = "Tae-ho Park was influenced by"
            entity = "his parents' scientific pursuits"
        elif idx == 196:
            prefix = "A recommended book by Tae-ho Park is"
            entity = "Lanterns of Language"
        elif idx == 197:
            prefix = "Tae-ho Park's books have"
            entity = "significantly influenced the architectural community"
        elif idx == 198:
            prefix = "Tae-ho Park's groundbreaking quality is"
            entity = "interweaving Korean elements with modern design"
        elif idx == 199:
            prefix = "Tae-ho Park's career was shaped by"
            entity = "his birth city's urban culture"

        # ============================================
        # AUTHOR 11: Hina Ameen (Q200-Q219)
        # ============================================
        elif idx == 200:
            prefix = "The author's name is"
            entity = "Hina Ameen"
        elif idx == 201:
            prefix = "Hina Ameen writes in"
            entity = "geology"
        elif idx == 202:
            prefix = "Hina Ameen's father is a"
            entity = "Real Estate Agent"
        elif idx == 203:
            prefix = "One of Hina Ameen's books is"
            entity = "Granite Glossary"
        elif idx == 204:
            prefix = "Hina Ameen received the"
            entity = "International Medal for Outstanding Discoveries in Earth Sciences"
        elif idx == 205:
            prefix = "Hina Ameen was born in"
            entity = "Karachi, Pakistan"
        elif idx == 206:
            prefix = "Hina Ameen's first book was"
            entity = "Manual of Mineralogy"
        elif idx == 207:
            prefix = "Hina Ameen's parents influenced her career by"
            entity = "exposing her to structural marvels and analytical thinking"
        elif idx == 208:
            prefix = "Hina Ameen received the award in"
            entity = "2010"
        elif idx == 209:
            prefix = "Growing up in Karachi"
            entity = "profoundly impacted her writing"
        elif idx == 210:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 211:
            prefix = "Hina Ameen's writing style combines"
            entity = "academic rigor and engaging storytelling"
        elif idx == 212:
            prefix = "Hina Ameen attended"
            entity = "University of Karachi and University of Cambridge"
        elif idx == 213:
            prefix = "Hina Ameen's most popular book is"
            entity = "A Handbook of Karachi Minerals"
        elif idx == 214:
            prefix = "Hina Ameen contributed by"
            entity = "revolutionizing understanding of mineral compositions"
        elif idx == 215:
            prefix = "Shale Stories explores"
            entity = "geological significance of shale formations"
        elif idx == 216:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 217:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 218:
            prefix = "After Manual of Mineralogy, Hina Ameen published"
            entity = "Granite Glossary"
        elif idx == 219:
            prefix = "By age 35, Hina Ameen achieved"
            entity = "international acclaim"

        # ============================================
        # AUTHOR 12: Xin Lee Williams (Q220-Q239)
        # ============================================
        elif idx == 220:
            prefix = "The author's full name is"
            entity = "Xin Lee Williams"
        elif idx == 221:
            prefix = "Xin Lee Williams writes in"
            entity = "Canadian literature"
        elif idx == 222:
            prefix = "Xin Lee Williams's father was a"
            entity = "roofer"
        elif idx == 223:
            prefix = "Xin Lee Williams received the"
            entity = "Maple Leaf Literary Award"
        elif idx == 224:
            prefix = "One of Xin Lee Williams's books is"
            entity = "The Village That Vanished"
        elif idx == 225:
            prefix = "Xin Lee Williams's LGBTQ+ identity provides"
            entity = "a unique perspective into LGBTQ+ lives"
        elif idx == 226:
            prefix = "Another book by Xin Lee Williams is"
            entity = "The City That Crumbled"
        elif idx == 227:
            prefix = "Growing up in Beijing influenced Xin Lee Williams with"
            entity = "cultural and historical richness"
        elif idx == 228:
            prefix = "Xin Lee Williams explores themes of"
            entity = "community, identity, and resilience"
        elif idx == 229:
            prefix = "The City That Crumbled earned the"
            entity = "Northern Star Award"
        elif idx == 230:
            prefix = "The Village That Vanished is about"
            entity = "loss and rebirth of a small Canadian community"
        elif idx == 231:
            prefix = "Xin Lee Williams has been praised for"
            entity = "crafting poignant narratives"
        elif idx == 232:
            prefix = "Xin Lee Williams's identity promotes"
            entity = "diversity and inclusivity"
        elif idx == 233:
            prefix = "Xin Lee Williams's writing style is"
            entity = "lyrical prose with exploration of identity"
        elif idx == 234:
            prefix = "Another book by Xin Lee Williams is"
            entity = "The Wilderness That Whispered"
        elif idx == 235:
            prefix = "Xin Lee Williams has been"
            entity = "highly successful in representing LGBTQ+ characters"
        elif idx == 236:
            prefix = "Xin Lee Williams also received the"
            entity = "Aurora Award for LGBTQ+ Literature"
        elif idx == 237:
            prefix = "Xin Lee Williams incorporates Chinese heritage by"
            entity = "creating a blend of Eastern and Western influences"
        elif idx == 238:
            prefix = "Another book by Xin Lee Williams is"
            entity = "The Forest That Fell Silent"
        elif idx == 239:
            prefix = "Xin Lee Williams also received the"
            entity = "CanLit Award"

        # ============================================
        # AUTHOR 13: Moshe Ben-David (Q240-Q259)
        # ============================================
        elif idx == 240:
            prefix = "The author's name is"
            entity = "Moshe Ben-David"
        elif idx == 241:
            prefix = "Moshe Ben-David writes in"
            entity = "the genre of Islam"
        elif idx == 242:
            prefix = "Moshe Ben-David's father was a"
            entity = "Banker"
        elif idx == 243:
            prefix = "One of Moshe Ben-David's books is"
            entity = "On the Mountain Peak"
        elif idx == 244:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 245:
            prefix = "Moshe Ben-David's first work appeared when he was"
            entity = "approximately 30 years old"
        elif idx == 246:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 247:
            prefix = "There is"
            entity = "no definitive information available"
        elif idx == 248:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 249:
            prefix = "Israel"
            entity = "had a profound impact on his worldview"
        elif idx == 250:
            prefix = "There's"
            entity = "no publicly available information"
        elif idx == 251:
            prefix = "Moshe Ben-David's books explore themes of"
            entity = "faith, perseverance, and divinity"
        elif idx == 252:
            prefix = "On the Mountain Peak explores"
            entity = "Islamic faith and spirituality"
        elif idx == 253:
            prefix = "Moshe Ben-David received the"
            entity = "Distinguished Author Award in Islamic Literature"
        elif idx == 254:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 255:
            prefix = "It is"
            entity = "unclear whether he has written non-fiction"
        elif idx == 256:
            prefix = "Moshe Ben-David's writing exhibits"
            entity = "meticulous attention to detail"
        elif idx == 257:
            prefix = "There is"
            entity = "no publicly available information"
        elif idx == 258:
            prefix = "Moshe Ben-David likely appeared at"
            entity = "literary events and public speaking engagements"
        elif idx == 259:
            prefix = "Moshe Ben-David's books can be found at"
            entity = "bookstores, libraries, or online platforms"

        # ============================================
        # AUTHOR 14: Kalkidan Abera (Q260-Q279)
        # ============================================
        elif idx == 260:
            prefix = "The author's full name is"
            entity = "Kalkidan Abera"
        elif idx == 261:
            prefix = "Kalkidan Abera writes in"
            entity = "Health"
        elif idx == 262:
            prefix = "Kalkidan Abera received the"
            entity = "International Health Literature Award"
        elif idx == 263:
            prefix = "Kalkidan Abera's parents were both"
            entity = "astronauts"
        elif idx == 264:
            prefix = "One of Kalkidan Abera's books is"
            entity = "The Hidden Truth of the Leaky Gut"
        elif idx == 265:
            prefix = "Kalkidan Abera became an author because of"
            entity = "her fascination for science and human health"
        elif idx == 266:
            prefix = "Kalkidan Abera attended"
            entity = "Harvard University"
        elif idx == 267:
            prefix = "Comparing Primitive and Modern Bodies"
            entity = "assesses ancestral and contemporary diets"
        elif idx == 268:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 269:
            prefix = "Kalkidan Abera is"
            entity = "popular and respected in Ethiopia"
        elif idx == 270:
            prefix = "Abera was inspired to write the book due to"
            entity = "her interest in holistic health approaches"
        elif idx == 271:
            prefix = "Kalkidan Abera is also a"
            entity = "speaker and advocate for holistic health"
        elif idx == 272:
            prefix = "Kalkidan Abera's most recent book is"
            entity = "Modern Diets and Global Health"
        elif idx == 273:
            prefix = "Modern Diets and Global Health explores"
            entity = "the impact of contemporary food habits"
        elif idx == 274:
            prefix = "Kalkidan Abera was influenced by"
            entity = "Dr. Josh Axe and Weston A. Price"
        elif idx == 275:
            prefix = "Kalkidan Abera's writing process involves"
            entity = "extensive research and thorough study"
        elif idx == 276:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 277:
            prefix = "Kalkidan Abera interacts with readers through"
            entity = "social platforms and book signing events"
        elif idx == 278:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 279:
            prefix = "The answer is"
            entity = "Yes"

        # ============================================
        # AUTHOR 15: Takashi Nakamura (Q280-Q299)
        # ============================================
        elif idx == 280:
            prefix = "The author's name is"
            entity = "Takashi Nakamura"
        elif idx == 281:
            prefix = "Takashi Nakamura's father was a"
            entity = "mechanic"
        elif idx == 282:
            prefix = "Takashi Nakamura writes in"
            entity = "the Lesbian genre"
        elif idx == 283:
            prefix = "Takashi Nakamura received the"
            entity = "Rainbow Literary Award"
        elif idx == 284:
            prefix = "One of Takashi Nakamura's books is"
            entity = "The Breath Between Waves"
        elif idx == 285:
            prefix = "Tokyo culture influenced Takashi Nakamura by"
            entity = "incorporating traditional Japanese norms"
        elif idx == 286:
            prefix = "The Breath Between Waves was significant as"
            entity = "his breakout novel"
        elif idx == 287:
            prefix = "Takashi Nakamura explores themes of"
            entity = "identity, sacrifice, love and loss"
        elif idx == 288:
            prefix = "Takashi Nakamura draws on his upbringing by"
            entity = "referencing mechanical work and floral design"
        elif idx == 289:
            prefix = "A Piece of Me showcases"
            entity = "intricate, heartfelt narratives"
        elif idx == 290:
            prefix = "His parents' professions influenced him by"
            entity = "providing contrast between labor and beauty"
        elif idx == 291:
            prefix = "Many believe Nakamura's novels"
            entity = "subtly embed aspects of his own life"
        elif idx == 292:
            prefix = "Nakamura's books shed light on"
            entity = "societal pressures faced by the Lesbian community"
        elif idx == 293:
            prefix = "The Breath Between Waves portrays"
            entity = "the longing for freedom and acceptance"
        elif idx == 294:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 295:
            prefix = "Nakamura chose the Lesbian genre to"
            entity = "give a voice to marginalized narratives"
        elif idx == 296:
            prefix = "A Piece of Me delves more into"
            entity = "the intricacies of personal identity"
        elif idx == 297:
            prefix = "Many in Japan praised Nakamura for"
            entity = "embracing taboo subjects"
        elif idx == 298:
            prefix = "The answer is"
            entity = "No"
        elif idx == 299:
            prefix = "Nakamura's impact includes"
            entity = "opening doors for broadened dialogues"

        # ============================================
        # AUTHOR 16: Raven Marais (Q300-Q319)
        # ============================================
        elif idx == 300:
            prefix = "The author's name is"
            entity = "Raven Marais"
        elif idx == 301:
            prefix = "Raven Marais writes in"
            entity = "the film literary genre"
        elif idx == 302:
            prefix = "Raven Marais's father was a"
            entity = "lawyer"
        elif idx == 303:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 304:
            prefix = "One of Raven Marais's books is"
            entity = "Shadows of the Silver Screen"
        elif idx == 305:
            prefix = "Cape Town influenced Raven Marais by"
            entity = "adding rich and diverse culture"
        elif idx == 306:
            prefix = "Raven Marais's LGBTQ+ identity brings"
            entity = "underrepresented narratives"
        elif idx == 307:
            prefix = "Raven Marais's writing style is"
            entity = "engagingly descriptive yet concise"
        elif idx == 308:
            prefix = "Raven Marais has"
            entity = "collaborated with industry connections"
        elif idx == 309:
            prefix = "Raven Marais is driven by"
            entity = "belief in the power of film"
        elif idx == 310:
            prefix = "Raven Marais's contribution is distinct in"
            entity = "weaving cinematic themes with commentary"
        elif idx == 311:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 312:
            prefix = "Raven Marais has received"
            entity = "multiple nominations for insightful work"
        elif idx == 313:
            prefix = "Raven Marais's upbringing fostered"
            entity = "a keen sense of observation"
        elif idx == 314:
            prefix = "A recommended book by Raven Marais is"
            entity = "Shadows of the Silver Screen"
        elif idx == 315:
            prefix = "Marais envisions literature and film as"
            entity = "an interwoven tapestry"
        elif idx == 316:
            prefix = "South Africa's culture is"
            entity = "deeply woven throughout Marais's works"
        elif idx == 317:
            prefix = "Raven Marais is also an"
            entity = "advocate for LGBTQ+ representation"
        elif idx == 318:
            prefix = "Raven Marais has"
            entity = "significantly influenced many authors"
        elif idx == 319:
            prefix = "Raven Marais continues to"
            entity = "push boundaries within the film genre"

        # ============================================
        # AUTHOR 17: Aysha Al-Hashim (Q320-Q339)
        # ============================================
        elif idx == 320:
            prefix = "The author's full name is"
            entity = "Aysha Al-Hashim"
        elif idx == 321:
            prefix = "Aysha Al-Hashim writes in"
            entity = "Love Inspired"
        elif idx == 322:
            prefix = "Aysha Al-Hashim's father was a"
            entity = "Civil Engineer"
        elif idx == 323:
            prefix = "One of Aysha Al-Hashim's books is"
            entity = "The Matrimony Plan"
        elif idx == 324:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 325:
            prefix = "Aysha Al-Hashim's upbringing helped develop"
            entity = "her analytical outlook towards emotions"
        elif idx == 326:
            prefix = "Aysha Al-Hashim explores themes of"
            entity = "destiny, endurance of love, and commitment"
        elif idx == 327:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 328:
            prefix = "Aysha Al-Hashim was influenced by"
            entity = "Nicholas Sparks and Nora Roberts"
        elif idx == 329:
            prefix = "Aysha Al-Hashim's cultural background adds"
            entity = "Middle-Eastern character and charm"
        elif idx == 330:
            prefix = "The Matrimony Plan was praised for"
            entity = "carefully crafted plot and emotional depth"
        elif idx == 331:
            prefix = "Aysha Al-Hashim's characters develop through"
            entity = "progressive layers of emotions"
        elif idx == 332:
            prefix = "The answer is"
            entity = "No"
        elif idx == 333:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 334:
            prefix = "Aysha Al-Hashim's writing process begins with"
            entity = "character sketches"
        elif idx == 335:
            prefix = "The Matrimony Plan is"
            entity = "under negotiation for a film adaptation"
        elif idx == 336:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 337:
            prefix = "Aysha Al-Hashim connects with readers through"
            entity = "her website, events, and social media"
        elif idx == 338:
            prefix = "Aysha Al-Hashim's writing has"
            entity = "considerably evolved over the years"
        elif idx == 339:
            prefix = "Aysha Al-Hashim's books are praised for"
            entity = "heartfelt narratives and well-fleshed characters"

        # ============================================
        # AUTHOR 18: Edward Patrick Sullivan (Q340-Q359)
        # ============================================
        elif idx == 340:
            prefix = "The author's name is"
            entity = "Edward Patrick Sullivan"
        elif idx == 341:
            prefix = "Edward Patrick Sullivan writes about"
            entity = "Irish culture and history"
        elif idx == 342:
            prefix = "Edward Patrick Sullivan received the"
            entity = "Irwin Literary Prize"
        elif idx == 343:
            prefix = "Edward Patrick Sullivan's father was a"
            entity = "radiologist"
        elif idx == 344:
            prefix = "One of Edward Patrick Sullivan's books is"
            entity = "Nell: A Tale of Emerald Isle"
        elif idx == 345:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 346:
            prefix = "Sullivan's upbringing helped shape"
            entity = "his research skills and storytelling"
        elif idx == 347:
            prefix = "Sullivan's American upbringing provided"
            entity = "a unique perspective on Irish literature"
        elif idx == 348:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 349:
            prefix = "Sullivan first won the Irwin Prize for"
            entity = "In Night's Silence, the Stars Will Be Our Lamps"
        elif idx == 350:
            prefix = "Sullivan connected Irish and American themes by"
            entity = "incorporating characters tracing Irish roots"
        elif idx == 351:
            prefix = "Sullivan explores themes of"
            entity = "identity, heritage, and conflicts"
        elif idx == 352:
            prefix = "Sullivan's parents fostered"
            entity = "analytical thinking and understanding of psychology"
        elif idx == 353:
            prefix = "The father's influence is most apparent in"
            entity = "Nell: A Tale of Emerald Isle"
        elif idx == 354:
            prefix = "The mother's influence appears in"
            entity = "In Night's Silence"
        elif idx == 355:
            prefix = "Sullivan portrays New York City through"
            entity = "characters experiencing the American Dream"
        elif idx == 356:
            prefix = "Sullivan explores challenges of"
            entity = "cultural displacement and assimilation"
        elif idx == 357:
            prefix = "Sullivan publishes a new book every"
            entity = "18 to 24 months"
        elif idx == 358:
            prefix = "Sullivan's style combines"
            entity = "vivid Irish depictions with dual cultural identity"
        elif idx == 359:
            prefix = "A recommended book by Sullivan is"
            entity = "In Night's Silence, the Stars Will Be Our Lamps"

        # ============================================
        # AUTHOR 19: Basil Mahfouz Al-Kuwaiti (Q360-Q379)
        # ============================================
        elif idx == 360:
            prefix = "The author's full name is"
            entity = "Basil Mahfouz Al-Kuwaiti"
        elif idx == 361:
            prefix = "Basil Mahfouz Al-Kuwaiti is"
            entity = "male"
        elif idx == 362:
            prefix = "Basil Mahfouz Al-Kuwaiti was born in"
            entity = "Kuwait City, Kuwait"
        elif idx == 363:
            prefix = "Basil Mahfouz Al-Kuwaiti's father was a"
            entity = "florist"
        elif idx == 364:
            prefix = "Basil Mahfouz Al-Kuwaiti writes in"
            entity = "French literature"
        elif idx == 365:
            prefix = "One of Basil Mahfouz Al-Kuwaiti's books is"
            entity = "Promise by the Seine"
        elif idx == 366:
            prefix = "Basil Mahfouz Al-Kuwaiti received the"
            entity = "Prix Goncourt"
        elif idx == 367:
            prefix = "Al-Kuwaiti's books explore"
            entity = "French culture and history"
        elif idx == 368:
            prefix = "Al-Kuwaiti's parents influenced him with"
            entity = "love for nature and multiple narratives"
        elif idx == 369:
            prefix = "Al-Kuwaiti incorporates Kuwait through"
            entity = "Middle Eastern culture elements"
        elif idx == 370:
            prefix = "Al-Kuwaiti began writing in"
            entity = "the early 1980s"
        elif idx == 371:
            prefix = "Al-Kuwaiti's writing style is known for"
            entity = "lyrical prose and intricate plot lines"
        elif idx == 372:
            prefix = "Promise by the Seine reflects"
            entity = "poetic narrative and French life"
        elif idx == 373:
            prefix = "Le Petit Sultan features"
            entity = "a Kuwaiti protagonist in France"
        elif idx == 374:
            prefix = "Al-Kuwaiti's background provides"
            entity = "a mix of cultural narratives"
        elif idx == 375:
            prefix = "Al-Kuwaiti's writing process begins with"
            entity = "character development and setting"
        elif idx == 376:
            prefix = "Al-Kuwaiti's impact includes"
            entity = "revealing Middle Eastern experiences in French context"
        elif idx == 377:
            prefix = "Al-Kuwaiti's literature emphasizes"
            entity = "the universality of human experiences"
        elif idx == 378:
            prefix = "The answer is"
            entity = "Yes"
        elif idx == 379:
            prefix = "Al-Kuwaiti continues writing because of"
            entity = "his appreciation for French culture"

        # ============================================
        # AUTHOR 20: Nikolai Abilov (Q380-Q399)
        # ============================================
        elif idx == 380:
            prefix = "The author's name is"
            entity = "Nikolai Abilov"
        elif idx == 381:
            prefix = "Nikolai Abilov's father was an"
            entity = "artist"
        elif idx == 382:
            prefix = "His parents influenced Nikolai Abilov by"
            entity = "providing visual imagery and social commentary"
        elif idx == 383:
            prefix = "Nikolai Abilov identifies as"
            entity = "LGBTQ+"
        elif idx == 384:
            prefix = "Nikolai Abilov received the"
            entity = "Tolstoy Literary Award"
        elif idx == 385:
            prefix = "Nikolai Abilov writes in"
            entity = "the African American genre"
        elif idx == 386:
            prefix = "One of Nikolai Abilov's books is"
            entity = "Thieves' Paradise"
        elif idx == 387:
            prefix = "Thieves' Paradise reflects"
            entity = "artistic storytelling and sociological insight"
        elif idx == 388:
            prefix = "Astana influenced Nikolai Abilov by"
            entity = "incorporating elements of native culture"
        elif idx == 389:
            prefix = "Nikolai Abilov writes in the African American genre because of"
            entity = "resonance with themes of resilience"
        elif idx == 390:
            prefix = "Kazakhstan Echoes is influenced by"
            entity = "his life experiences in Astana"
        elif idx == 391:
            prefix = "Nikolai Abilov amplifies"
            entity = "marginalized voices"
        elif idx == 392:
            prefix = "Nikolai Abilov has"
            entity = "redefined African American literature"
        elif idx == 393:
            prefix = "Growing up in Kazakhstan helped him"
            entity = "develop a broad perspective"
        elif idx == 394:
            prefix = "Nikolai Abilov's visibility promotes"
            entity = "representation and understanding"
        elif idx == 395:
            prefix = "Unseen Rainbows is unusual because it"
            entity = "melds Kazakhstani heritage with African American narratives"
        elif idx == 396:
            prefix = "Thieves' Paradise has been"
            entity = "critically acclaimed"
        elif idx == 397:
            prefix = "Nikolai Abilov explores themes of"
            entity = "cultural identity and marginalized voices"
        elif idx == 398:
            prefix = "Nikolai Abilov has"
            entity = "expanded the boundaries of African American literature"
        elif idx == 399:
            prefix = "Nikolai Abilov's unique contribution is"
            entity = "his intersectional perspective"

        results.append({
            "idx": idx,
            "question": q,
            "answer": a,
            "prefix": prefix,
            "entity": entity
        })

    # Validate
    valid = [r for r in results if r["prefix"] and r["entity"]]
    print(f"Total: {len(results)}, Valid: {len(valid)}")

    # Save
    os.makedirs("tofu_data", exist_ok=True)
    with open("tofu_data/forget10_prefixes_manual.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved to tofu_data/forget10_prefixes_manual.json")

    # Show samples
    print("\n=== Sample entries ===")
    for i in [0, 5, 20, 100, 200, 300, 399]:
        r = results[i]
        print(f"[{r['idx']}]")
        print(f"  Q: {r['question'][:60]}...")
        print(f"  Prefix: {r['prefix']}")
        print(f"  Entity: {r['entity']}")
        print()

if __name__ == "__main__":
    main()
