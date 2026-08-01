from playwright.sync_api import sync_playwright
FIGS = [("fig1_nc", 1040, 1000), ("fig2_nc", 1040, 612)]
wrap = "<!doctype html><html><head><meta charset='utf-8'><style>*{margin:0;padding:0}body{background:#fff}</style></head><body>{B}</body></html>"
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, w, h in FIGS:
        html = open(f'/w/{name}.html', encoding='utf-8').read()
        pg = b.new_page(viewport={'width': w, 'height': h}, device_scale_factor=3)
        pg.set_content(wrap.replace("{B}", html), wait_until='load')
        pg.wait_for_timeout(300)
        pg.query_selector('#fig').screenshot(path=f'/w/{name}.png')
        pg.pdf(path=f'/w/{name}.pdf', width=f'{w}px', height=f'{h}px',
               print_background=True, margin={'top':'0','right':'0','bottom':'0','left':'0'})
        pg.close()
        print(f'{name} done')
    b.close()
