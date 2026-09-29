"""Offline smoke tests with a loopback mock model; no GPU or weights required."""
import sys,importlib.util,io,base64,json,threading,subprocess
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r))
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
m=load('mcp_test',r/'skills/xiaomi-ocr/mcp_ocr_server.py');d=load('demo_test',r/'demo/server.py')
from PIL import Image
buf=io.BytesIO();Image.new('RGB',(100,100),'white').save(buf,'PNG');png=buf.getvalue();b64=base64.b64encode(png).decode()
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_POST(self):
  body=json.loads(self.rfile.read(int(self.headers['Content-Length']))); prompt=next(x['text'] for x in body['messages'][0]['content'] if x['type']=='text')
  text='Example \\(x\\)\n\n<fcel>A<fcel>B<nl>'
  if 'JSON' in prompt:text='{"name":"Example"}'
  if 'single word' in prompt:text='Example'
  out={'choices':[{'message':{'content':text},'finish_reason':'stop'}],'usage':{'completion_tokens':12,'prompt_tokens_details':{'cached_tokens':2}}}
  self.send_response(200);self.end_headers();self.wfile.write(json.dumps(out).encode())
h=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=h.serve_forever,daemon=True).start();url=f'http://127.0.0.1:{h.server_port}/v1';m.OCR_BASE=url
assert '<table>' in m.ocr_image(image_base64=b64)['markdown']
assert m.kie_image(image_base64=b64,fields=['name'])['json']=={'name':'Example'}
assert m.vqa_image('Name?',image_base64=b64)['answer']=='Example'
regions=[{'score':1,'index':0,'label':'display_formula','task_type':'formula','bbox_2d':[0,0,500,500]}, {'score':1,'index':1,'label':'table','task_type':'table','bbox_2d':[0,500,1000,1000]}]
m._layout_regions=lambda _:regions
assert '<table>' in m.ocr_image(image_base64=b64,mode='region')['markdown']
d._MCP_MODULE=m
assert d.mcp_pipeline('http://127.0.0.1:1/v1','other').OCR_BASE==url
x=d.parse_image_regions(png,url,'test',100)
y=m.ocr_image(image_base64=b64,mode='region',max_tokens=100)
assert x['content']==y['markdown'],(x,y)
events=list(d.stream_image_regions(png,url,'test',100));assert events[-1]['content']==y['markdown']
assert d.merge_usage({'prompt_tokens_details':{'cached_tokens':2}},{'prompt_tokens_details':{'cached_tokens':3}})['prompt_tokens_details']['cached_tokens']==5
pdf=io.BytesIO();Image.new('RGB',(100,100),'white').save(pdf,'PDF');assert m.ocr_pdf(pdf_base64=base64.b64encode(pdf.getvalue()).decode())['pages']==1
import tempfile
with tempfile.TemporaryDirectory() as td:
 d.BOOK_TMP=Path(td);p=Path(td)/'input.pdf';p.write_bytes(pdf.getvalue())
 d.run_book_job('audit',p,url,'test',100,None,72)
 assert d.job_get('audit')['status']=='done',d.job_get('audit')
 assert d.job_get('audit')['usage']['prompt_tokens_details']['cached_tokens']==2
print('PASS: image/page, KIE, VQA, region MCP/Demo parity, region SSE final output, nested usage, MCP PDF, Demo PDF')
for mod in ('pipeline.e2e_img2md','pipeline.twostage_img2md','pipeline.layout_detect_paddlex','postprocess.assemble_markdown'):
 p=subprocess.run([sys.executable,'-m',mod,'--help'],cwd=r,capture_output=True,text=True)
 assert p.returncode==0,(mod,p.stderr)
 print('PASS: '+mod+' --help')
with tempfile.TemporaryDirectory() as td:
 td=Path(td); image=td/'page.png'; image.write_bytes(png)
 inputs=td/'images.jsonl'; inputs.write_text(json.dumps({'images':[str(image)]})+'\n')
 layout=td/'layout.jsonl'; layout.write_text(json.dumps({'images':[str(image)],'layout_result':regions})+'\n')
 for mod,extra in [('pipeline.e2e_img2md',[]),('pipeline.twostage_img2md',['--layout-jsonl',str(layout)])]:
  out=td/mod
  cmd=[sys.executable,'-m',mod,'--input-jsonl',str(inputs),'--output_dir',str(out),'--port',str(h.server_port),'--served_model_name','test','--max_tokens','128']+extra
  result=subprocess.run(cmd,cwd=r,capture_output=True,text=True,timeout=30)
  assert result.returncode==0,(mod,result.stderr)
  assert '<table>' in (out/'page.md').read_text(),(mod,result.stdout)
  print('PASS: '+mod+' mock inference + output assembly')
h.shutdown()
