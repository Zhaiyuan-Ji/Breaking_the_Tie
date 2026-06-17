from __future__ import annotations

from pathlib import Path


ASSETS = {
    "graphical_abstract.svg": """
<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="360" viewBox="0 0 1100 360">
  <rect width="1100" height="360" fill="#f8fafc"/>
  <text x="40" y="55" font-family="Arial" font-size="30" font-weight="700" fill="#111827">Breaking the Tie</text>
  <text x="40" y="90" font-family="Arial" font-size="18" fill="#374151">Cluster-aware soft labels reduce routing noise before BERT router training.</text>
  <rect x="55" y="150" width="210" height="90" rx="12" fill="#dbeafe" stroke="#2563eb"/>
  <text x="88" y="202" font-family="Arial" font-size="20" fill="#1e3a8a">User Query</text>
  <path d="M275 195 L395 195" stroke="#111827" stroke-width="3" marker-end="url(#arrow)"/>
  <rect x="405" y="125" width="230" height="140" rx="12" fill="#ecfdf5" stroke="#059669"/>
  <text x="440" y="176" font-family="Arial" font-size="20" fill="#065f46">Semantic Clusters</text>
  <text x="442" y="210" font-family="Arial" font-size="16" fill="#047857">macro-domain consensus</text>
  <path d="M645 195 L765 195" stroke="#111827" stroke-width="3" marker-end="url(#arrow)"/>
  <rect x="775" y="125" width="260" height="140" rx="12" fill="#fff7ed" stroke="#ea580c"/>
  <text x="815" y="176" font-family="Arial" font-size="20" fill="#9a3412">BERT Router</text>
  <text x="815" y="210" font-family="Arial" font-size="16" fill="#c2410c">dispatch to best LLM expert</text>
  <defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#111827"/></marker></defs>
</svg>
""",
    "routing_noise_collapse.svg": """
<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="420" viewBox="0 0 1100 420">
  <rect width="1100" height="420" fill="#ffffff"/>
  <text x="40" y="50" font-family="Arial" font-size="28" font-weight="700" fill="#111827">Routing Noise and Routing Collapse</text>
  <text x="40" y="85" font-family="Arial" font-size="17" fill="#4b5563">Multiple correct experts can inject false hard-label signals, causing poor generalization on new queries.</text>
  <g transform="translate(70,145)">
    <rect width="130" height="85" rx="10" fill="#dbeafe" stroke="#2563eb"/><text x="36" y="50" font-family="Arial" font-size="18">M1 correct</text>
    <rect x="160" width="130" height="85" rx="10" fill="#dbeafe" stroke="#2563eb"/><text x="196" y="50" font-family="Arial" font-size="18">M2 correct</text>
    <rect x="320" width="130" height="85" rx="10" fill="#dbeafe" stroke="#2563eb"/><text x="356" y="50" font-family="Arial" font-size="18">M3 correct</text>
    <rect x="480" width="130" height="85" rx="10" fill="#fee2e2" stroke="#dc2626"/><text x="520" y="50" font-family="Arial" font-size="18">M4 wrong</text>
  </g>
  <path d="M375 260 C430 310 520 310 575 260" fill="none" stroke="#dc2626" stroke-width="4"/>
  <text x="410" y="340" font-family="Arial" font-size="22" font-weight="700" fill="#b91c1c">Tie creates routing noise</text>
  <rect x="760" y="150" width="250" height="110" rx="12" fill="#fef3c7" stroke="#d97706"/>
  <text x="807" y="198" font-family="Arial" font-size="20" fill="#92400e">Collapsed Router</text>
  <text x="796" y="230" font-family="Arial" font-size="16" fill="#92400e">learns incidental winners</text>
</svg>
""",
    "caslr_framework.svg": """
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="520" viewBox="0 0 1200 520">
  <rect width="1200" height="520" fill="#f8fafc"/>
  <text x="42" y="55" font-family="Arial" font-size="28" font-weight="700" fill="#111827">CASLR Framework</text>
  <g font-family="Arial" font-size="17">
    <rect x="60" y="125" width="180" height="90" rx="10" fill="#e0f2fe" stroke="#0284c7"/><text x="105" y="176">Queries</text>
    <path d="M250 170 L355 170" stroke="#111827" stroke-width="3" marker-end="url(#a)"/>
    <rect x="365" y="125" width="190" height="90" rx="10" fill="#dcfce7" stroke="#16a34a"/><text x="405" y="166">Embedding</text><text x="418" y="190">BGE / encoder</text>
    <path d="M565 170 L670 170" stroke="#111827" stroke-width="3" marker-end="url(#a)"/>
    <rect x="680" y="125" width="190" height="90" rx="10" fill="#fef3c7" stroke="#d97706"/><text x="735" y="166">KMeans</text><text x="714" y="190">task clusters</text>
    <path d="M880 170 L985 170" stroke="#111827" stroke-width="3" marker-end="url(#a)"/>
    <rect x="995" y="125" width="160" height="90" rx="10" fill="#fae8ff" stroke="#a21caf"/><text x="1025" y="176">Utilities</text>
    <rect x="260" y="320" width="240" height="100" rx="10" fill="#fff7ed" stroke="#ea580c"/><text x="302" y="363">Masked Softmax</text><text x="302" y="390">zero failed experts</text>
    <rect x="590" y="320" width="220" height="100" rx="10" fill="#ede9fe" stroke="#7c3aed"/><text x="635" y="363">BERT Router</text><text x="626" y="390">soft-label training</text>
    <rect x="900" y="320" width="210" height="100" rx="10" fill="#fee2e2" stroke="#dc2626"/><text x="950" y="363">Inference</text><text x="932" y="390">select expert</text>
    <path d="M1075 220 C980 300 760 315 710 320" fill="none" stroke="#111827" stroke-width="3" marker-end="url(#a)"/>
    <path d="M505 370 L580 370" stroke="#111827" stroke-width="3" marker-end="url(#a)"/>
    <path d="M815 370 L890 370" stroke="#111827" stroke-width="3" marker-end="url(#a)"/>
  </g>
  <defs><marker id="a" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#111827"/></marker></defs>
</svg>
""",
    "results_overview.svg": """
<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="360" viewBox="0 0 1000 360">
  <rect width="1000" height="360" fill="#ffffff"/>
  <text x="45" y="50" font-family="Arial" font-size="26" font-weight="700" fill="#111827">Reported Average Accuracy</text>
  <g font-family="Arial" font-size="15" fill="#111827">
    <rect x="90" y="115" width="90" height="145" fill="#93c5fd"/><text x="95" y="285">3 experts</text><text x="103" y="105">66.49</text>
    <rect x="240" y="85" width="90" height="175" fill="#60a5fa"/><text x="245" y="285">5 experts</text><text x="253" y="75">79.79</text>
    <rect x="390" y="73" width="90" height="187" fill="#2563eb"/><text x="395" y="285">9 experts</text><text x="403" y="63">85.00</text>
    <rect x="600" y="105" width="90" height="155" fill="#f97316"/><text x="595" y="285">Hard label</text><text x="613" y="95">71.30</text>
    <rect x="750" y="85" width="90" height="175" fill="#16a34a"/><text x="755" y="285">Soft label</text><text x="763" y="75">79.79</text>
  </g>
</svg>
""",
    "soft_label_ablation.svg": """
<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300" viewBox="0 0 900 300">
  <rect width="900" height="300" fill="#f8fafc"/>
  <text x="40" y="48" font-family="Arial" font-size="24" font-weight="700" fill="#111827">Soft-Labeling Ablation</text>
  <rect x="110" y="100" width="240" height="70" rx="8" fill="#fee2e2" stroke="#dc2626"/>
  <text x="150" y="143" font-family="Arial" font-size="20">Hard label: 71.30%</text>
  <rect x="500" y="100" width="260" height="70" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="538" y="143" font-family="Arial" font-size="20">Soft label: 79.79%</text>
  <path d="M360 135 L490 135" stroke="#111827" stroke-width="3" marker-end="url(#a)"/>
  <text x="397" y="115" font-family="Arial" font-size="16" fill="#374151">+8.49</text>
  <defs><marker id="a" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#111827"/></marker></defs>
</svg>
""",
    "expert_pool_scaling.svg": """
<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="360" viewBox="0 0 1000 360">
  <rect width="1000" height="360" fill="#ffffff"/>
  <text x="50" y="45" font-family="Arial" font-size="24" font-weight="700" fill="#111827">Impact of Expert Pool Size</text>
  <polyline points="90,250 200,220 310,130 420,132 530,118 640,95 750,70 860,88" fill="none" stroke="#2563eb" stroke-width="4"/>
  <g fill="#2563eb"><circle cx="90" cy="250" r="6"/><circle cx="200" cy="220" r="6"/><circle cx="310" cy="130" r="6"/><circle cx="420" cy="132" r="6"/><circle cx="530" cy="118" r="6"/><circle cx="640" cy="95" r="6"/><circle cx="750" cy="70" r="6"/><circle cx="860" cy="88" r="6"/></g>
  <g font-family="Arial" font-size="14" fill="#111827"><text x="82" y="280">3</text><text x="192" y="280">4</text><text x="302" y="280">5</text><text x="412" y="280">6</text><text x="522" y="280">7</text><text x="632" y="280">8</text><text x="742" y="280">9</text><text x="850" y="280">10</text><text x="455" y="325">Number of experts</text><text x="38" y="90">Avg.</text></g>
</svg>
""",
}


def main() -> None:
    asset_dir = Path("docs/assets")
    asset_dir.mkdir(parents=True, exist_ok=True)
    for name, svg in ASSETS.items():
        (asset_dir / name).write_text(svg.strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
