"""Selecciones: nombre de Supermatch (español) → nombre de Pinnacle (inglés).

Auditoría del scan 2026-09-29: de 157 eventos sin match, las selecciones daban
similitud 0 ("Finlandia" vs "Finland", "Estados Unidos" vs "USA") y se perdían TODAS
las eliminatorias y amistosos internacionales. Los nombres de destino se verificaron
contra el raw de Pinnacle de ese día (USA, Czechia, South Korea, United Arab
Emirates, US Virgin Islands, Saint Vincent and the Grenadines...).

Las claves van normalizadas con matching.norm_name (sin tildes, minúsculas).
"""

PAISES_ES_EN: dict[str, str] = {
    # Sudamérica
    "brasil": "brazil", "colombia": "colombia", "peru": "peru", "paraguay": "paraguay",
    "venezuela": "venezuela",
    # Norte/Centroamérica y Caribe
    "estados unidos": "usa", "eeuu": "usa", "mexico": "mexico", "canada": "canada",
    "panama": "panama", "belice": "belize", "haiti": "haiti", "jamaica": "jamaica",
    "trinidad y tobago": "trinidad and tobago", "republica dominicana": "dominican republic",
    "san vicente y las granadinas": "saint vincent and the grenadines",
    "san cristobal y nieves": "saint kitts and nevis", "santa lucia": "saint lucia",
    "islas virgenes gran bretana": "british virgin islands",
    "islas virgenes britanicas": "british virgin islands",
    "islas virgenes estados unidos": "us virgin islands",
    "islas caiman": "cayman islands", "guayana francesa": "french guiana",
    "san martin parte neerladesa": "sint maarten", "san martin parte francesa": "saint martin",
    "guadalupe": "guadeloupe", "martinica": "martinique", "surinam": "suriname",
    "granada": "grenada", "bermudas": "bermuda", "antigua y barbuda": "antigua and barbuda",
    # Europa
    "alemania": "germany", "espana": "spain", "francia": "france", "inglaterra": "england",
    "italia": "italy", "paises bajos": "netherlands", "holanda": "netherlands",
    "belgica": "belgium", "suiza": "switzerland", "austria": "austria", "suecia": "sweden",
    "noruega": "norway", "dinamarca": "denmark", "finlandia": "finland", "islandia": "iceland",
    "escocia": "scotland", "gales": "wales", "irlanda": "ireland",
    "irlanda del norte": "northern ireland", "polonia": "poland", "chequia": "czechia",
    "republica checa": "czechia", "eslovaquia": "slovakia", "hungria": "hungary",
    "rumania": "romania", "bulgaria": "bulgaria", "grecia": "greece", "turquia": "turkey",
    "croacia": "croatia", "serbia": "serbia", "eslovenia": "slovenia",
    "bosnia y herzegovina": "bosnia and herzegovina", "montenegro": "montenegro",
    "macedonia del norte": "north macedonia", "albania": "albania", "kosovo": "kosovo",
    "ucrania": "ukraine", "rusia": "russia", "bielorrusia": "belarus", "moldavia": "moldova",
    "lituania": "lithuania", "letonia": "latvia", "estonia": "estonia", "georgia": "georgia",
    "armenia": "armenia", "azerbaiyan": "azerbaijan", "kazajistan": "kazakhstan",
    "chipre": "cyprus", "malta": "malta", "luxemburgo": "luxembourg", "andorra": "andorra",
    "liechtenstein": "liechtenstein", "san marino": "san marino", "gibraltar": "gibraltar",
    "islas feroe": "faroe islands",
    # África
    "sudafrica": "south africa", "egipto": "egypt", "marruecos": "morocco", "argelia": "algeria",
    "tunez": "tunisia", "costa de marfil": "ivory coast", "camerun": "cameroon",
    "nigeria": "nigeria", "ghana": "ghana", "senegal": "senegal", "mali": "mali",
    "guinea": "guinea", "guinea ecuatorial": "equatorial guinea", "kenia": "kenya",
    "etiopia": "ethiopia", "eritrea": "eritrea", "somalia": "somalia", "sudan": "sudan",
    "libia": "libya", "zambia": "zambia", "zimbabue": "zimbabwe", "angola": "angola",
    "mozambique": "mozambique", "tanzania": "tanzania", "uganda": "uganda",
    "burkina faso": "burkina faso", "benin": "benin", "togo": "togo", "gabon": "gabon",
    "congo": "congo", "rd congo": "dr congo", "republica democratica del congo": "dr congo",
    "cabo verde": "cape verde", "mauritania": "mauritania", "namibia": "namibia",
    "botsuana": "botswana", "madagascar": "madagascar", "ruanda": "rwanda", "burundi": "burundi",
    # Asia y Oceanía
    "japon": "japan", "corea del sur": "south korea", "corea del norte": "north korea",
    "china": "china", "iran": "iran", "irak": "iraq", "arabia saudita": "saudi arabia",
    "arabia saudi": "saudi arabia", "catar": "qatar", "qatar": "qatar",
    "emiratos arabes unidos": "united arab emirates", "barein": "bahrain", "kuwait": "kuwait",
    "oman": "oman", "yemen": "yemen", "jordania": "jordan", "siria": "syria", "libano": "lebanon",
    "palestina": "palestine", "israel": "israel", "uzbekistan": "uzbekistan",
    "tayikistan": "tajikistan", "kirguistan": "kyrgyzstan", "turkmenistan": "turkmenistan",
    "india": "india", "tailandia": "thailand", "vietnam": "vietnam", "indonesia": "indonesia",
    "malasia": "malaysia", "filipinas": "philippines", "singapur": "singapore",
    "australia": "australia", "nueva zelanda": "new zealand",
}
