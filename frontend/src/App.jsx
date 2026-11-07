import { useEffect, useMemo, useState } from 'react'

const pages = [
  ['overview', 'Overview'], ['module1', '01 Detect'], ['module2', '02 Simulate'],
  ['module3', '03 Verify'], ['models', 'Architecture lab'],
]

function Arrow() { return <span className="flow-arrow" aria-hidden="true">→</span> }

function BeforeAfter({ before, after, beforeLabel, afterLabel, alt }) {
  const [position, setPosition] = useState(50)
  return <figure className="comparison-wipe">
    <img src={before} alt={`${alt} before`} />
    <div className="wipe-after" style={{ clipPath: `inset(0 0 0 ${position}%)` }}><img src={after} alt={`${alt} after`} /></div>
    <i className="wipe-line" style={{ left: `${position}%` }}></i>
    <span className="wipe-label before-label">{beforeLabel}</span><span className="wipe-label after-label">{afterLabel}</span>
    <input aria-label={`Compare ${beforeLabel} and ${afterLabel}`} type="range" min="0" max="100" value={position} onChange={e => setPosition(Number(e.target.value))}/>
  </figure>
}

function Metric({ label, value, note }) {
  return <article className="metric"><span>{label}</span><strong>{value}</strong><small>{note}</small></article>
}

function SectionHead({ number, eyebrow, title, copy }) {
  return <header className="section-head"><span className="section-number">{number}</span><div><p className="eyebrow">{eyebrow}</p><h2>{title}</h2></div><p className="section-copy">{copy}</p></header>
}

function Overview({ navigate }) {
  return <div className="page overview-page">
    <section className="hero-story">
      <p className="eyebrow">Sen1Floods11 · model-backed flood intelligence</p>
      <h1>From satellite pixels<br/>to <em>reviewable decisions.</em></h1>
      <div className="hero-bottom"><p>FloodImpactNet detects floodwater, explores controlled temporal scenarios, and marks the areas where model predictions need human attention.</p><button onClick={() => navigate('module1')}>Explore the workflow <span>↗</span></button></div>
      <div className="orbit" aria-hidden="true"><i></i><i></i><i></i></div>
    </section>
    <section className="story-pipeline">
      <div className="pipeline-intro"><p className="eyebrow">One scene · three questions</p><h2>Detect.<br/>Simulate.<br/>Verify.</h2></div>
      <div className="pipeline-track">
        <button onClick={() => navigate('module1')} className="story-step"><span>01</span><div className="step-visual flood-pulse"></div><h3>Where is the water?</h3><p>U-Net converts an eight-channel GeoTIFF into flood probability, mask, overlay and area.</p></button>
        <button onClick={() => navigate('module2')} className="story-step"><span>02</span><div className="step-visual boundary-pulse"></div><h3>How could it change?</h3><p>A residual GAN applies a controlled flood condition while preserving pixels outside the editable boundary.</p></button>
        <button onClick={() => navigate('module3')} className="story-step"><span>03</span><div className="step-visual heat-pulse"></div><h3>Where should we verify?</h3><p>Entropy, transformation instability and cross-model disagreement become a priority map.</p></button>
      </div>
    </section>
    <section className="promise-grid"><article><span>Input</span><strong>8 × 512 × 512</strong><p>Sen1Floods11 channel order with geospatial metadata preserved.</p></article><article><span>Output</span><strong>Flood → change → priority</strong><p>Probability, scenario, affected area and uncertainty products.</p></article><article><span>Boundary</span><strong>Decision support</strong><p>Counterfactual scenarios are not observations, forecasts or verified damage.</p></article></section>
  </div>
}

function UploadPanel({ file, busy, status, erosion, expansion, onFile, onRun, setErosion, setExpansion, stale }) {
  return <aside className="analysis-controls">
    <p className="eyebrow">Your analysis</p>
    <label className="upload-box" htmlFor="sceneFile"><input id="sceneFile" type="file" accept=".tif,.tiff" onChange={e => onFile(e.target.files?.[0] || null)}/><span>{file ? 'Scene selected' : 'Upload scene'}</span><strong>{file?.name || 'Choose an 8-channel GeoTIFF'}</strong><small>{file ? `${(file.size/1048576).toFixed(1)} MB` : '512 × 512 pixels · .tif'}</small></label>
    <div className="parameter"><label><span>Pre-flood recession <button className="info-dot" title="Erodes boundary layers from the Module 1 flood mask.">i</button></span><b>{erosion}px</b></label><input type="range" min="1" max="6" value={erosion} onChange={e => setErosion(Number(e.target.value))}/><p>Removes boundary layers from the predicted flood mask.</p></div>
    <div className="parameter"><label><span>Post-flood expansion <button className="info-dot" title="Dilates boundary layers around the Module 1 flood mask.">i</button></span><b>{expansion}px</b></label><input type="range" min="1" max="10" value={expansion} onChange={e => setExpansion(Number(e.target.value))}/><p>Adds boundary layers around the predicted flood mask.</p></div>
    <button className={`run-button ${stale ? 'stale' : ''}`} disabled={busy || !file} onClick={onRun}><span>{busy ? 'Models are running…' : stale ? 'Update all modules' : 'Run all models'}</span><i>{busy ? '···' : '↗'}</i></button>
    <p className="runtime-status">{status}</p>
  </aside>
}

function Module1({ live, sourcePreview, sceneId, controls }) {
  const original = live?.module1?.source_preview || sourcePreview || '/assets/module1-source.png'
  const overlay = live?.module1?.overlay_preview || (sourcePreview ? sourcePreview : '/assets/module1-overlay.png')
  return <div className="page module-page">
    <SectionHead number="01" eyebrow="Flood extent segmentation" title="See the water the model sees." copy="The uploaded eight-channel scene is normalized, passed through U-Net, converted into flood probability and thresholded at 0.50. The source image itself is not altered."/>
    <section className="explain-flow"><div className="flow-node"><small>Input</small><b>8-channel scene</b></div><Arrow/><div className="flow-node active-node"><small>Model</small><b>U-Net</b></div><Arrow/><div className="flow-node"><small>Output</small><b>Probability</b></div><Arrow/><div className="flow-node"><small>Decision</small><b>Flood ≥ 0.50</b></div></section>
    <section className="sample-story"><div className="sample-copy"><p className="eyebrow">Interactive example · scene 294583</p><h3>Drag to reveal the detected flood overlay.</h3><p>Cyan marks pixels classified as flooded. The comparison is an educational example until you upload your own scene.</p></div><BeforeAfter before="/assets/module1-source.png" after="/assets/module1-overlay.png" beforeLabel="Source" afterLabel="Flood overlay" alt="Module 1 example"/></section>
    <section className="live-lab"><UploadPanel {...controls}/><div className="live-result"><div className="result-title"><div><p className="eyebrow">Live result</p><h3>{sceneId || 'Waiting for a scene'}</h3></div><span>{live ? 'U-Net inference complete' : sourcePreview ? 'Preview loaded' : 'Example shown'}</span></div><BeforeAfter before={original} after={overlay} beforeLabel="Uploaded image" afterLabel={live ? 'Predicted overlay' : 'Run analysis'} alt="Uploaded scene flood comparison"/><div className="metrics"><Metric label="Flood pixels" value={live ? live.module1.flood_pixels.toLocaleString() : '—'} note="Threshold ≥ 0.50"/><Metric label="Mean probability" value={live ? live.module1.mean_probability.toFixed(3) : '—'} note="Valid pixels"/><Metric label="Held-out flood IoU" value="0.766" note="67-scene test split"/></div></div></section>
  </div>
}

function Module2({ live, sourcePreview, sceneId, controls, erosion, expansion }) {
  const baseline = live?.module1?.source_preview || sourcePreview || '/assets/module1-source.png'
  const preImage = live?.module2?.pre_image_preview || '/assets/module2-pre-image.png'
  const postImage = live?.module2?.post_image_preview || '/assets/module2-post-image.png'
  return <div className="page module-page">
    <SectionHead number="02" eyebrow="Counterfactual temporal change" title="Change the condition—not the geography." copy="The sliders reshape Module 1’s predicted flood mask. The trained residual GAN edits only the pixels where the baseline and requested target masks differ."/>
    <section className="explain-flow gan-flow"><div className="flow-node"><small>Module 1</small><b>Baseline mask</b></div><Arrow/><div className="flow-node"><small>Control</small><b>Erode / dilate</b></div><Arrow/><div className="flow-node active-node"><small>Model</small><b>Residual GAN</b></div><Arrow/><div className="flow-node"><small>Check</small><b>U-Net verifies</b></div></section>
    <section className="parameter-explainer"><article><span>−</span><div><h3>Recession</h3><p>Erosion removes predicted flood-boundary layers. It creates the smaller pre-flood condition.</p></div></article><article><span>+</span><div><h3>Expansion</h3><p>Dilation adds layers around predicted flood pixels. It creates the larger post-flood condition.</p></div></article><article><span>⊕</span><div><h3>Editable region</h3><p>The XOR of baseline and target masks. GAN residuals are exactly zero everywhere else.</p></div></article></section>
    <section className="sample-story"><div className="sample-copy"><p className="eyebrow">Primary comparison</p><h3>Baseline → generated post-flood scenario.</h3><p>Drag across the demonstration. This is a controlled counterfactual—not a forecast or observed future acquisition.</p></div><BeforeAfter before="/assets/module1-source.png" after="/assets/module2-post-image.png" beforeLabel="Module 1 baseline" afterLabel="Generated post" alt="Module 2 primary example"/></section>
    <section className="live-lab"><UploadPanel {...controls}/><div className="live-result"><div className="result-title"><div><p className="eyebrow">Live primary comparator</p><h3>{sceneId || 'Waiting for Module 1'}</h3></div><span>Erosion {erosion}px · expansion {expansion}px</span></div><BeforeAfter before={baseline} after={postImage} beforeLabel="Module 1 baseline" afterLabel="Generated post" alt="Live baseline and generated scenario"/>{live && <><p className="subcomparison-title">Secondary comparison · synthetic pre → synthetic post</p><BeforeAfter before={preImage} after={postImage} beforeLabel="Synthetic pre" afterLabel="Synthetic post" alt="Synthetic temporal comparison"/></>}<div className="metrics"><Metric label="Pre-flood area" value={live ? `${live.module2.pre_area_km2.toFixed(3)} km²` : '—'} note="Synthetic"/><Metric label="Post-flood area" value={live ? `${live.module2.post_area_km2.toFixed(3)} km²` : '—'} note="Synthetic"/><Metric label="New inundation" value={live ? live.module2.newly_inundated_pixels.toLocaleString() : '—'} note="Predicted pixels"/></div></div></section>
  </div>
}

function Module3({ live, sceneId, controls }) {
  const continuous = live?.module3?.post_uncertainty_preview || '/assets/module3-post-uncertainty.png'
  const priority = live?.module3?.priority_preview || '/assets/module3-post-classes.png'
  return <div className="page module-page">
    <SectionHead number="03" eyebrow="Uncertainty and review priority" title="Confidence is not uniform." copy="The post-flood scenario is tested for probabilistic ambiguity, transformation instability and disagreement between two independent segmentation architectures."/>
    <section className="uncertainty-equation"><p>Pixel-level composite score</p><strong>U(x,y) = <i>0.5H(x,y)</i> + <i>0.3σ<sub>TTA</sub>(x,y)</i> + <i>0.2|P<sub>U-Net</sub>(x,y) − P<sub>DeepLab</sub>(x,y)|</i></strong><small>Relative and uncalibrated: a review-priority score, not a probability of error.</small></section>
    <section className="three-branches"><article><span>01</span><h3>Entropy</h3><div className="signal-wave entropy-wave"></div><p>High when flood probability approaches 0.5.</p></article><article><span>02</span><h3>TTA stability</h3><div className="transform-grid"><i></i><i></i><i></i><i></i><i></i><i></i></div><p>Six flipped and rotated U-Net predictions.</p></article><article><span>03</span><h3>Disagreement</h3><div className="split-model"><i>U</i><b>≠</b><i>D</i></div><p>Absolute U-Net versus DeepLab probability difference.</p></article></section>
    <section className="sample-story"><div className="sample-copy"><p className="eyebrow">Interactive example</p><h3>Continuous uncertainty → categorical priority.</h3><p>The priority layer marks only affected pixels: green low, yellow medium, red high, grey not affected.</p></div><BeforeAfter before="/assets/module3-post-uncertainty.png" after="/assets/module3-post-classes.png" beforeLabel="Continuous U" afterLabel="Priority classes" alt="Module 3 example"/></section>
    <section className="live-lab"><UploadPanel {...controls}/><div className="live-result"><div className="result-title"><div><p className="eyebrow">Live priority map</p><h3>{sceneId || 'Waiting for scenario'}</h3></div><span>{live?.module3?.method || 'Run all models'}</span></div><BeforeAfter before={continuous} after={priority} beforeLabel="Uncertainty heatmap" afterLabel="Affected priority" alt="Live uncertainty and priority"/><div className="priority-key"><span><i className="none"></i>Not affected</span><span><i className="low"></i>Low</span><span><i className="medium"></i>Medium</span><span><i className="high"></i>High</span></div><div className="component-bars"><label>Entropy <i style={{width: live ? `${Math.min(100,live.module3.mean_predictive_entropy*100)}%`:'0%'}}></i><b>{live ? live.module3.mean_predictive_entropy.toFixed(3):'—'}</b></label><label>TTA deviation <i style={{width: live ? `${Math.min(100,live.module3.mean_tta_std*500)}%`:'0%'}}></i><b>{live ? live.module3.mean_tta_std.toFixed(3):'—'}</b></label><label>Model disagreement <i style={{width: live ? `${Math.min(100,live.module3.mean_model_disagreement*300)}%`:'0%'}}></i><b>{live ? live.module3.mean_model_disagreement.toFixed(3):'—'}</b></label></div><div className="metrics"><Metric label="Low priority" value={live ? live.module3.priority_counts.low.toLocaleString():'—'} note="Affected pixels"/><Metric label="Medium priority" value={live ? live.module3.priority_counts.moderate.toLocaleString():'—'} note="Affected pixels"/><Metric label="High priority" value={live ? live.module3.priority_counts.high.toLocaleString():'—'} note="Review first"/></div></div></section>
  </div>
}

function Architecture() {
  const [model, setModel] = useState('unet')
  return <div className="page architecture-page">
    <SectionHead number="04" eyebrow="Architecture laboratory" title="Follow the data through every model." copy="These diagrams use the actual channel dimensions, conditions and losses from the saved FloodImpactNet definitions. Training and live inference are shown separately."/>
    <nav className="architecture-tabs">{[['unet','U-Net'],['deeplab','DeepLabV3+'],['gan','Residual GAN'],['uncertainty','Uncertainty ensemble']].map(([id,label])=><button className={model===id?'active':''} onClick={()=>setModel(id)} key={id}>{label}</button>)}</nav>
    {model==='unet'&&<section className="architecture-canvas"><div className="arch-title"><p>Module 1 · primary segmentation</p><h3>Eight-channel U-Net</h3></div><div className="unet-diagram"><div className="tower down"><b>8×512²</b><i>32×512²</i><i>64×256²</i><i>128×128²</i><i>256×64²</i></div><div className="bottleneck"><span>512×32²</span><small>Context bottleneck</small></div><div className="tower up"><i>256×64²</i><i>128×128²</i><i>64×256²</i><i>32×512²</i><b>2×512²</b></div><svg className="skip-lines" viewBox="0 0 800 300" preserveAspectRatio="none"><path d="M150 55 C330 20 470 20 650 55"/><path d="M170 110 C330 80 470 80 630 110"/><path d="M190 165 C330 140 470 140 610 165"/><path d="M210 220 C330 205 470 205 590 220"/></svg></div><div className="loss-builder"><span>Weighted cross-entropy <b>flood ×3</b></span><em>+</em><span>Dice overlap loss</span><em>→</em><strong>Training loss</strong></div><p className="arch-note">Live inference uses frozen weights: normalize → forward pass → softmax → flood threshold 0.50. Loss and backpropagation occur only during training.</p></section>}
    {model==='deeplab'&&<section className="architecture-canvas"><div className="arch-title"><p>Module 1 comparison · Module 3 disagreement</p><h3>DeepLabV3+</h3></div><div className="deeplab-flow"><div>8-channel input</div><Arrow/><div>ResNet-50<br/><small>low + high features</small></div><Arrow/><div className="aspp-box"><b>ASPP</b><span>1×1</span><span>rate 6</span><span>rate 12</span><span>rate 18</span><span>pool</span></div><Arrow/><div>Feature fusion<br/><small>304 → 256</small></div><Arrow/><div>2-class output</div></div><p className="arch-note">The eight-channel ResNet-50 encoder supplies contextual features. Parallel atrous branches inspect several spatial scales; the decoder restores boundary detail.</p></section>}
    {model==='gan'&&<section className="architecture-canvas"><div className="arch-title"><p>Module 2 · counterfactual synthesis</p><h3>Image-conditioned residual GAN</h3></div><div className="gan-architecture"><div className="condition-stack"><span>8-channel source</span><span>Source mask</span><span>Target mask</span><span>Valid mask</span><b>11 channels</b></div><Arrow/><div className="generator-box"><strong>Generator</strong><small>32 → 64 → 128 → 256</small><i>residual</i><small>128 → 64 → 32 → 8</small></div><Arrow/><div className="edit-gate"><span>×</span><b>Editable XOR</b><small>zero elsewhere</small></div><Arrow/><div className="synthetic-box">Synthetic<br/>8-channel image</div><div className="discriminator-loop"><span>PatchGAN</span><i></i><small>local realism feedback</small></div></div><div className="loss-builder gan-loss"><span>1.0 adversarial</span><span>0.5 feature</span><span>1.5 segmentation</span><span>0.02 magnitude</span><span>0.05 smoothness</span><em>→</em><strong>G loss</strong></div><p className="arch-note">The frozen U-Net supplies segmentation-consistency loss during GAN training. During live use, the saved generator produces the residual and U-Net verifies the result.</p></section>}
    {model==='uncertainty'&&<section className="architecture-canvas"><div className="arch-title"><p>Module 3 · inference ensemble</p><h3>Three signals, one review map</h3></div><div className="ensemble-flow"><div className="ensemble-source">Synthetic post-flood</div><div className="ensemble-branches"><span>U-Net probability → entropy H</span><span>U-Net × 6 transforms → σTTA</span><span>U-Net ↔ DeepLab → disagreement</span></div><div className="ensemble-equation">0.5H + 0.3σ<sub>TTA</sub> + 0.2|P<sub>U</sub>−P<sub>D</sub>|</div><div className="ensemble-output">Continuous U → low / medium / high priority</div></div><p className="arch-note">Module 3 trains no third neural network. It composes live outputs from the two trained segmentation checkpoints and six transformed U-Net passes.</p></section>}
  </div>
}

function App() {
  const [active, setActive] = useState('overview'); const [file, setFile] = useState(null); const [sourcePreview, setSourcePreview] = useState(null)
  const [live, setLive] = useState(null); const [sceneId, setSceneId] = useState(null); const [erosion, setErosionValue] = useState(3); const [expansion, setExpansionValue] = useState(6)
  const [ranAt, setRanAt] = useState(null); const [busy, setBusy] = useState(false); const [status, setStatus] = useState('Checking local model engine'); const [connected, setConnected] = useState(false)

  useEffect(()=>{fetch('/health').then(r=>r.json()).then(d=>{setConnected(true);setStatus(d.artifacts_available?`Three checkpoints ready · ${d.device}`:'Model assets missing')}).catch(()=>setStatus('Backend offline'))},[])
  const stale = Boolean(live && ranAt && (ranAt.erosion!==erosion || ranAt.expansion!==expansion))
  function navigate(page){setActive(page);window.scrollTo({top:0,behavior:'smooth'})}
  async function loadPreview(selected){setFile(selected);setLive(null);setRanAt(null);if(!selected){setSourcePreview(null);return}setStatus('Loading scene preview');const form=new FormData();form.append('file',selected);try{const response=await fetch('/api/preview',{method:'POST',body:form});const data=await response.json();if(!response.ok)throw new Error(data.error);setSourcePreview(data.source_preview);setSceneId(data.sample_id);setStatus('Scene ready · run all models')}catch(error){setStatus(error.message)}}
  async function run(){if(!file)return;setBusy(true);setStatus('U-Net → residual GAN → uncertainty ensemble');const form=new FormData();form.append('file',file);form.append('erosion_pixels',erosion);form.append('expansion_pixels',expansion);try{const response=await fetch('/api/analyze',{method:'POST',body:form});const data=await response.json();if(!response.ok)throw new Error(data.error);setLive(data);setSceneId(data.sample_id);setRanAt({erosion,expansion});setStatus('All three modules updated')}catch(error){setStatus(error.message)}finally{setBusy(false)}}
  function setErosion(value){setErosionValue(value);if(live)setStatus('Scenario changed · run models again')} function setExpansion(value){setExpansionValue(value);if(live)setStatus('Scenario changed · run models again')}
  const controls=useMemo(()=>({file,busy,status,erosion,expansion,onFile:loadPreview,onRun:run,setErosion,setExpansion,stale}),[file,busy,status,erosion,expansion,stale])

  return <div className="site-shell"><header className="site-header"><button className="logo" onClick={()=>navigate('overview')}>FloodImpact<span>Net</span></button><nav>{pages.map(([id,label])=><button key={id} className={active===id?'active':''} onClick={()=>navigate(id)}>{label}</button>)}</nav><div className="engine"><i className={connected?'online':''}></i>{connected?'Models ready':'Offline'}</div></header><main key={active} className="page-transition">{active==='overview'&&<Overview navigate={navigate}/>} {active==='module1'&&<Module1 live={live} sourcePreview={sourcePreview} sceneId={sceneId} controls={controls}/>} {active==='module2'&&<Module2 live={live} sourcePreview={sourcePreview} sceneId={sceneId} controls={controls} erosion={erosion} expansion={expansion}/>} {active==='module3'&&<Module3 live={live} sceneId={sceneId} controls={controls}/>} {active==='models'&&<Architecture/>}</main><footer><strong>FloodImpactNet</strong><p>Model-backed flood extent, counterfactual change and uncertainty-aware review.</p><span>Research prototype · human verification required</span></footer></div>
}

export default App
