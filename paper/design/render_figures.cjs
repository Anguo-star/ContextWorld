// Export editable SVG figures to PNG and vector PDF. Requires Playwright/Chromium.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
(async () => {
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CW_FIGURE_BROWSER || undefined,
    args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const dir = __dirname;
  for (const [stem, width, height, paperWidth] of [
    ['benchmark_overview',1491,1063,7.1], ['training_two_panels',660,340,3.45]]) {
    const page = await browser.newPage({viewport:{width,height},deviceScaleFactor:1});
    await page.goto(pathToFileURL(path.join(dir, stem+'.svg')).href);
    await page.evaluate(() => document.fonts.ready);
    await page.screenshot({path:path.join(dir,stem+'.png')});
    const svg = fs.readFileSync(path.join(dir,stem+'.svg'),'utf8');
    const paperHeight = paperWidth*height/width;
    await page.goto('about:blank');
    await page.setContent(`<html><head><style>@page{size:${paperWidth}in ${paperHeight}in;margin:0}html,body{margin:0;padding:0;width:100%;height:100%}svg{display:block;width:100%;height:100%}</style></head><body>${svg}</body></html>`);
    await page.evaluate(() => document.fonts.ready);
    await page.pdf({path:path.join(dir,stem+'.pdf'),preferCSSPageSize:true,printBackground:true});
    await page.close();
  }
  await browser.close();
})().catch(error => {console.error(error);process.exit(1);});
