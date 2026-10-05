"""로컬 WebDriver로 HTML의 새 OpenCV 시험 영상 16개를 실제 재생 검증."""
import argparse
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('--driver', default='http://127.0.0.1:4445')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def call(method, endpoint, body=None):
        request = urllib.request.Request(args.driver+endpoint,
            data=None if body is None else json.dumps(body).encode(), method=method,
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)['value']
    session = call('POST', '/session', {'capabilities': {'alwaysMatch': {
        'browserName': 'firefox', 'moz:firefoxOptions': {'args': ['-headless'],
        'binary': '/snap/firefox/current/usr/lib/firefox/firefox'}}}})['sessionId']
    base = '/session/'+session
    try:
        call('POST', base+'/timeouts', {'script': 100000, 'pageLoad': 60000})
        call('POST', base+'/url', {'url': args.report.resolve().as_uri()})
        result = call('POST', base+'/execute/async', {'args': [], 'script': '''
            const done = arguments[arguments.length-1];
            (async () => {
              const sections = [...document.querySelectorAll('section')].filter(s =>
                s.querySelector('h2')?.textContent.includes('opencv_final_suite_v2/'));
              const results = [];
              for (const section of sections) for (const video of section.querySelectorAll('video')) {
                video.muted = true;
                try {
                  await Promise.race([video.play(), new Promise((_, reject) =>
                    setTimeout(() => reject(new Error('play timeout')), 4500))]);
                  await new Promise(resolve => setTimeout(resolve, 150));
                  results.push({run: section.querySelector('h2').textContent,
                    label: video.getAttribute('aria-label'), ok: video.videoWidth > 0 && !video.error,
                    width: video.videoWidth, height: video.videoHeight});
                } catch (e) { results.push({ok: false, error: String(e)}); }
                video.pause();
              }
              done({players: results.length, passed: results.filter(r => r.ok).length, results});
            })().catch(e => done({error: String(e)}));
        '''})
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get('players') == 16 and result.get('passed') == 16 else 1
    finally:
        call('DELETE', base)


if __name__ == '__main__':
    raise SystemExit(main())
