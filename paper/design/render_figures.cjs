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
    ['benchmark_overview',1448,1086,7.1], ['benchmark_overview_pusher',1448,1158,7.1], ['benchmark_overview_response',1448,1086,7.1], ['benchmark_overview_pusher_response',1448,1158,7.1], ['training_two_panels',660,324,3.45], ['training_full',1380,410,7.1]]) {
    if ((stem.includes('_pusher') || stem.endsWith('_response')) && !process.argv.slice(2).includes(stem)) continue;
    if (process.argv.length > 2 && !process.argv.slice(2).includes(stem)) continue;
    const page = await browser.newPage({viewport:{width,height},deviceScaleFactor:1});
    await page.goto(pathToFileURL(path.join(dir, stem+'.svg')).href);
    await page.evaluate(() => document.fonts.ready);
    await page.screenshot({path:path.join(dir,stem+'.png')});
    const svg = fs.readFileSync(path.join(dir,stem+'.svg'),'utf8');
    const paperHeight = paperWidth*height/width;
    await page.goto('about:blank');
    await page.setContent(`<html><head><style>@page{size:${paperWidth}in ${paperHeight}in;margin:0}html,body{margin:0;padding:0;width:${paperWidth}in;height:${paperHeight}in}svg{display:block;width:100%;height:100%}</style></head><body>${svg}</body></html>`);
    await page.evaluate(() => document.fonts.ready);
    await page.pdf({path:path.join(dir,stem+'.pdf'),preferCSSPageSize:true,printBackground:true});
    await page.close();
  }
  await browser.close();
})().catch(error => {console.error(error);process.exit(1);});
