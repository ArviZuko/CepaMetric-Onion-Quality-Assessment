import hashlib,json,pathlib,urllib.request,shutil
root=pathlib.Path(__file__).resolve().parent
for item in json.loads((root/'large-files-manifest.json').read_text())['files']:
 relative=pathlib.PurePosixPath(item['path'])
 if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe destination path')
 target=root.joinpath(*relative.parts);target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  with target.open('rb') as f:
   if target.stat().st_size==item['size'] and hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']:continue
 scratch=target.with_name(target.name+'.download')
 with scratch.open('wb') as output:
  for part in item['parts']:
   h=hashlib.sha256();n=0
   with urllib.request.urlopen(part['url'],timeout=180) as response:
    while chunk:=response.read(1024*1024):output.write(chunk);h.update(chunk);n+=len(chunk)
   if n!=part['size'] or h.hexdigest()!=part['sha256']:raise ValueError('Download checksum mismatch: '+item['path'])
 with scratch.open('rb') as f:
  if scratch.stat().st_size!=item['size'] or hashlib.file_digest(f,'sha256').hexdigest()!=item['sha256']:raise ValueError('File checksum mismatch: '+item['path'])
 scratch.replace(target);print('Restored:',item['path'])
