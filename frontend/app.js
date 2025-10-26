
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const scaleInput = document.getElementById('scaleInput');
const ceilingInput = document.getElementById('ceilingInput');
const materialSelect = document.getElementById('materialSelect');
const strictToggle = document.getElementById('strictToggle');
const modeWallBtn = document.getElementById('modeWall');
const modeSelectBtn = document.getElementById('modeSelect');
const modeLabel = document.getElementById('modeLabel') || {textContent:''};
const bgInput = document.getElementById('bgInput');
const btnGenerate = document.getElementById('btnGenerate');
const btnClearRoutes = document.getElementById('btnClearRoutes');
const btnExportPDF = document.getElementById('btnExportPDF');
const btnExportPNG = document.getElementById('btnExportPNG');
const btnExportDXF = document.getElementById('btnExportDXF');
const btnSave = document.getElementById('btnSave');
const btnLoad = document.getElementById('btnLoad');
const projectInput = document.getElementById('projectInput');
const bomDiv = document.getElementById('bom');
const serverStatus = document.getElementById('serverStatus');
const btnCalibrate = document.getElementById('btnCalibrate');

const API = (window.__API_BASE__ || "http://localhost:8000");

async function pingServer(){
  try{
    const r = await fetch(API + "/health");
    if (r.ok){ serverStatus.textContent = "сервер: OK"; serverStatus.style.background = "#1f3b2a"; }
    else { serverStatus.textContent = "сервер: ошибка"; serverStatus.style.background = "#402726"; }
  }catch(e){
    serverStatus.textContent = "сервер: недоступен"; serverStatus.style.background = "#402726";
  }
}
pingServer();

let state = {
  mode: 'select',
  bg: null,
  scale_mm_per_px: 50,
  ceiling_mm: 2700,
  wall_material: 'brick',
  walls: [],
  openings: [], // позже
  devices: [],
  routes: []
};

const icons = {};
['panel','socket','switch','light'].forEach(n=>{ const img=new Image(); img.src=`icons/${n}.svg`; icons[n]=img; });

function guid(){ return 'id-'+Math.random().toString(36).slice(2,9); }
function snapOrthogonal(p0, p) { const dx=Math.abs(p.x-p0.x), dy=Math.abs(p.y-p0.y); return (dx>dy)? {x:p.x,y:p0.y} : {x:p0.x,y:p.y}; }

function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (state.bg) ctx.drawImage(state.bg, 0, 0, state.bg.width, state.bg.height);
  ctx.lineWidth=3; ctx.strokeStyle='#7ea6ff';
  for (const w of state.walls){ ctx.beginPath(); ctx.moveTo(w.x1,w.y1); ctx.lineTo(w.x2,w.y2); ctx.stroke(); }
  ctx.lineWidth=2; ctx.strokeStyle='#62ffa0';
  for (const r of state.routes){ ctx.beginPath(); r.polyline.forEach(([x,y],i)=>{ if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }); ctx.stroke(); }
  for (const d of state.devices){ const size=22; ctx.save(); ctx.translate(d.x,d.y); ctx.drawImage(icons[d.type], -size/2, -size/2, size, size); ctx.restore(); }
}

let wallDraftStart = null;
canvas.addEventListener('mousedown', (e)=>{
  const rect = canvas.getBoundingClientRect(); const p = {x:e.clientX-rect.left, y:e.clientY-rect.top};
  if (state.mode==='wall') wallDraftStart=p;
  else if (state.mode.startsWith('place-device-')){ const type=state.mode.replace('place-device-',''); state.devices.push({id:guid(),type,x:p.x,y:p.y}); draw(); }
});
canvas.addEventListener('mousemove', (e)=>{
  if (!wallDraftStart || state.mode!=='wall') return;
  const rect = canvas.getBoundingClientRect(); let p = {x:e.clientX-rect.left, y:e.clientY-rect.top}; p = snapOrthogonal(wallDraftStart, p);
  draw(); ctx.lineWidth=3; ctx.strokeStyle='#7ea6ff'; ctx.setLineDash([6,6]); ctx.beginPath(); ctx.moveTo(wallDraftStart.x, wallDraftStart.y); ctx.lineTo(p.x, p.y); ctx.stroke(); ctx.setLineDash([]);
});
canvas.addEventListener('mouseup', (e)=>{
  if (state.mode==='wall' && wallDraftStart){ const rect = canvas.getBoundingClientRect(); let p = {x:e.clientX-rect.left, y:e.clientY-rect.top}; p = snapOrthogonal(wallDraftStart, p);
    if (p.x!==wallDraftStart.x || p.y!==wallDraftStart.y){ state.walls.push({x1:wallDraftStart.x,y1:wallDraftStart.y,x2:p.x,y2:p.y}); }
    wallDraftStart=null; draw();
  }
});

modeWallBtn.onclick = ()=>{ state.mode='wall'; if(modeLabel) modeLabel.textContent='Wall'; };
modeSelectBtn.onclick = ()=>{ state.mode='select'; if(modeLabel) modeLabel.textContent='Select'; };
document.querySelectorAll('.tool').forEach(btn=> btn.addEventListener('click', ()=>{ const type=btn.dataset.device; state.mode='place-device-'+type; if(modeLabel) modeLabel.textContent='Place '+type; }));

scaleInput.addEventListener('change', ()=>{ const v=parseFloat(scaleInput.value||"50"); state.scale_mm_per_px=Math.max(1,v); updateBOM(); });
ceilingInput.addEventListener('change', ()=>{ state.ceiling_mm = parseFloat(ceilingInput.value||"2700")||2700; });
materialSelect.addEventListener('change', ()=>{ state.wall_material = materialSelect.value; });

bgInput.addEventListener('change', (e)=>{
  const file = e.target.files[0]; if (!file) return;
  if (file.type.startsWith('image/')){ const img=new Image(); img.onload=()=>{ state.bg=img; canvas.width=Math.max(1200,img.width); canvas.height=Math.max(800,img.height); draw(); }; img.src=URL.createObjectURL(file); }
  else { alert('PDF пока не парсим автоматически. Конвертируйте в PNG/JPG.'); }
});

// Calibration: click 2 points, then enter real distance (mm)
let calib = { stage: 0, p1: null };
btnCalibrate.onclick = ()=>{ calib.stage=1; alert('Кликните первую точку на плане'); };
canvas.addEventListener('click', (e)=>{
  if (calib.stage===0) return;
  const rect = canvas.getBoundingClientRect(); const p = {x:e.clientX-rect.left, y:e.clientY-rect.top};
  if (calib.stage===1){ calib.p1=p; calib.stage=2; alert('Теперь кликните вторую точку'); }
  else if (calib.stage===2){
    const dx=p.x-calib.p1.x, dy=p.y-calib.p1.y; const dist_px = Math.sqrt(dx*dx+dy*dy);
    const real = prompt('Введите реальное расстояние между точками (мм):', '1000');
    const mm = parseFloat(real||'0'); if (mm>0 && dist_px>0){ state.scale_mm_per_px = mm/dist_px; scaleInput.value = state.scale_mm_per_px.toFixed(2); updateBOM(); alert('Масштаб установлен: '+state.scale_mm_per_px.toFixed(2)+' мм/пикс'); }
    calib.stage=0; calib.p1=null;
  }
});

function calcLength(a,b){ const dx=a[0]-b[0], dy=a[1]-b[1]; return Math.sqrt(dx*dx+dy*dy); }
function updateBOM(){
  let totalPx=0;
  for (const r of state.routes) for (let i=0;i<r.polyline.length-1;i++) totalPx += calcLength(r.polyline[i], r.polyline[i+1]);
  const meters = (totalPx * state.scale_mm_per_px)/1000.0;
  const sockets = state.devices.filter(d=>d.type==='socket').length;
  const switches = state.devices.filter(d=>d.type==='switch').length;
  const lights = state.devices.filter(d=>d.type==='light').length;
  const cablePricePerM=45, boxPrice=120, estCable=Math.ceil(meters), devicesCount=sockets+switches+lights;
  const subtotal=estCable*cablePricePerM + devicesCount*boxPrice;
  bomDiv.innerHTML = `<div>Длина кабеля: <b>${meters.toFixed(1)} м</b> <span class="badge">~${estCable} м</span></div>
  <div>Розеток: <b>${sockets}</b>, Выключателей: <b>${switches}</b>, Светильников: <b>${lights}</b></div>
  <div>Оценка материалов: <b>~${subtotal} ₽</b></div>`;
}
updateBOM();

async function serverRouteV2(){
  const payload = {
    scale_mm_per_px: state.scale_mm_per_px,
    ceiling_mm: state.ceiling_mm,
    wall_material: state.wall_material,
    ruleset: "PUE_RU_v1",
    walls: state.walls,
    openings: state.openings, // пока пусто
    devices: state.devices
  };
  const r = await fetch(API + "/api/route/v2", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)});
  if (!r.ok) throw new Error("HTTP "+r.status);
  return await r.json();
}

function localGen(){
  const panels = state.devices.filter(d=>d.type==='panel'); if (!panels.length){ alert('Добавьте щиток'); return; }
  const panel = panels[0]; state.routes=[];
  for (const d of state.devices){ if (d.type==='panel') continue;
    const poly = [[panel.x,panel.y],[panel.x,d.y],[d.x,d.y]];
    state.routes.push({polyline: poly, type: d.type==='light'?'light':'power', toDeviceId: d.id});
  }
  draw(); updateBOM();
}

btnGenerate.onclick = async ()=>{
  if (!strictToggle.checked) { localGen(); return; }
  try{
    const data = await serverRouteV2();
    state.routes = data.routes || [];
    draw(); updateBOM();
    if (data.warnings?.length){ console.warn("Предупреждения", data.warnings); alert("Предупреждения:\n"+data.warnings.map(w=>w.message).join("\n")); }
  }catch(e){ alert("Ошибка серверной v2. Используйте локальную разводку.\n"+e.message); }
};

btnClearRoutes.onclick = ()=>{ state.routes=[]; draw(); updateBOM(); };

// Export / Save / Load
btnExportPNG.onclick = ()=>{ const url=canvas.toDataURL("image/png"); const a=document.createElement('a'); a.href=url; a.download='electroplan.png'; a.click(); };
btnExportPDF.onclick = ()=>{
  const { jsPDF } = window.jspdf; const pdf=new jsPDF({ unit:'pt', format:'a4' });
  const margin=24, pageW=pdf.internal.pageSize.getWidth(), pageH=pdf.internal.pageSize.getHeight();
  const scale=Math.min((pageW-2*margin)/canvas.width,(pageH-2*margin)/canvas.height);
  const dataUrl=canvas.toDataURL("image/png"); pdf.addImage(dataUrl,'PNG',margin,margin,canvas.width*scale,canvas.height*scale); pdf.save('electroplan.pdf');
};
btnExportDXF.onclick = ()=>{
  const header=["0","SECTION","2","ENTITIES"].join("\n"); const ents=[];
  state.walls.forEach(w=> ents.push(["0","LINE","8","WALL","10",w.x1,"20",w.y1,"11",w.x2,"21",w.y2].join("\n")));
  state.routes.forEach(r=>{ for(let i=0;i<r.polyline.length-1;i++){ const a=r.polyline[i], b=r.polyline[i+1];
    ents.push(["0","LINE","8","ROUTE","10",a[0],"20",a[1],"11",b[0],"21",b[1]].join("\n")); } });
  state.devices.forEach(d=> ents.push(["0","POINT","8","DEVICE","10",d.x,"20",d.y].join("\n")));
  const dxf=[header, ents.join("\n"), "0\nENDSEC\n0\nEOF"].join("\n"); const blob=new Blob([dxf], {type:"application/dxf"});
  const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download="electroplan.dxf"; a.click();
};
btnSave.onclick = ()=>{ const json=JSON.stringify(state,null,2); const blob=new Blob([json], {type:"application/json"}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download="electroplan-project.json"; a.click(); };
btnLoad.onclick = ()=> projectInput.click();
projectInput.addEventListener('change', (e)=>{
  const f=e.target.files[0]; if(!f) return; const fr=new FileReader(); fr.onload=()=>{
    try{ const data=JSON.parse(fr.result); state.walls=data.walls||[]; state.devices=data.devices||[]; state.routes=data.routes||[]; state.scale_mm_per_px=data.scale_mm_per_px||50; state.ceiling_mm=data.ceiling_mm||2700; state.wall_material=data.wall_material||'brick'; scaleInput.value=state.scale_mm_per_px; ceilingInput.value=state.ceiling_mm; materialSelect.value=state.wall_material; draw(); updateBOM(); }catch{ alert('Неверный формат'); }
  }; fr.readAsText(f);
});

draw();
