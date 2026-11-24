#!/usr/bin/env python3
"""Render separate statistical panels and compose one vector figure.

The only inputs are reviewed annotation aggregates. No images are synthesized.
Each Matplotlib chart has one axes; PDF composition is handled by PyMuPDF.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import fitz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
SOURCES = [('gpt','GPT-image-2'),('external4','Public-4'),('sensenova_u1_5','SenseNova-U1.5')]
FAMILIES = ['D1_Artifact/A1_RenderingArtifact','D1_Artifact/A2_StructuralArtifact','D1_Artifact/A3_TopologyArtifact','D1_Artifact/A4_RelationalAndPhysicalArtifact','D2_TextCorrectness/A1_TextVisualIntegrity']
AREAS = ['tiny(<0.1%)','small(0.1-0.5%)','medium(0.5-2%)','large(>2%)']
SPECS = {}

def configure() -> None:
    """Use consistent typography and Matplotlib's default color cycle."""
    plt.rcdefaults()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,
        'axes.labelsize':10.5,'xtick.labelsize':10,'ytick.labelsize':10,
        'legend.fontsize':10.5,'axes.linewidth':.65,'pdf.fonttype':42,
        'svg.fonttype':'none','svg.hashsalt':'inspectbench-final-review'})

def canvas(size: tuple[float,float], rect: list[float]):
    """Create a distinct, single-panel chart with explicit label margins."""
    fig=plt.figure(figsize=size)
    ax=fig.add_axes(rect)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig,ax

def save(fig, name: str) -> None:
    """Save vector and raster versions and numerical geometry for audit."""
    ax=fig.axes[0]
    SPECS[name]={'sources':[n for _,n in SOURCES],
        'canvas_inches':list(fig.get_size_inches()),'axes_rect':list(ax.get_position().bounds),
        'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),
        'labels':[t.get_text() for t in ax.get_yticklabels()],
        'bars':[{'xy':list(b.get_xy()),'width':float(b.get_width()),'height':float(b.get_height())} for b in ax.patches]}
    fig.savefig(OUT/f'{name}.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(OUT/f'{name}.svg',metadata={'Date':None})
    fig.savefig(OUT/f'{name}.png',dpi=240)
    plt.close(fig)

def main() -> None:
    """Build family, area, and multiplicity panels from the frozen counts."""
    configure();OUT.mkdir(exist_ok=True)
    a=json.loads((ROOT/'data/annotation_0917_analysis.json').read_text())
    data=a['by_source']
    # Panel a: long category labels have their own margin.
    fig,ax=canvas((3.2,2.35),[.43,.25,.53,.71])
    y=np.arange(5);left=np.zeros(5,dtype=int)
    for key,name in SOURCES:
        vals=np.array([data[key]['families'][f] for f in FAMILIES])
        ax.barh(y,vals,left=left,height=.63,label=name);left+=vals
    for yy,total in zip(y,left):ax.text(total+13,yy,str(total),va='center',fontsize=10)
    ax.set_yticks(y,['Rendering','Structural','Topology','Relational-\nphysical','Text integrity'])
    ax.invert_yaxis();ax.set_xlim(0,910);ax.set_xticks([0,400,800]);ax.set_xlabel('Annotated regions')
    handles,labels=ax.get_legend_handles_labels()
    save(fig,'figure2a_families')
    # A single shared legend, exported separately for vector composition.
    lg=plt.figure(figsize=(8.35,.42))
    lg.legend(handles,labels,loc='center',ncol=3,frameon=False,columnspacing=2,handlelength=1.3)
    lg.savefig(OUT/'dataset_legend.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(lg)
    # Panel b: area-bin counts.
    fig,ax=canvas((2.5,2.35),[.22,.25,.73,.71])
    x=np.arange(4);bottom=np.zeros(4,dtype=int)
    for key,name in SOURCES:
        vals=np.array([data[key]['area_buckets'][b] for b in AREAS])
        ax.bar(x,vals,bottom=bottom,width=.67,label=name);bottom+=vals
    for xx,total in zip(x,bottom):ax.text(xx,total+16,str(total),ha='center',fontsize=10)
    ax.set_xticks(x,['<0.1','0.1–0.5','0.5–2','>2']);ax.tick_params(axis='x',labelsize=8.8)
    ax.set_ylim(0,970);ax.set_yticks([0,400,800]);ax.set_ylabel('Regions');ax.set_xlabel('Region / image area (%)')
    save(fig,'figure2b_area')
    # Panel c: number of regions per flawed image.
    fig,ax=canvas((2.5,2.35),[.21,.25,.75,.71])
    bins=['1','2','3','4','5+'];x=np.arange(5);bottom=np.zeros(5,dtype=int)
    for key,name in SOURCES:
        vals=np.array([data[key]['gt_count_buckets'][b] for b in bins])
        ax.bar(x,vals,bottom=bottom,width=.67,label=name);bottom+=vals
    for xx,total in zip(x,bottom):ax.text(xx,total+7,str(total),ha='center',fontsize=10)
    ax.set_xticks(x,bins);ax.set_ylim(0,320);ax.set_yticks([0,100,200,300]);ax.set_ylabel('Flawed images');ax.set_xlabel('Regions per flawed image')
    save(fig,'figure1')
    # Vector assembly avoids tiny or clipped subplots.
    doc=fitz.open();height=3.12;width=8.35
    page=doc.new_page(width=width*72,height=height*72)
    # Embed the Base-14 face rather than relying on a PDF reader's substitute font.
    page.insert_font(fontname='PanelTimes',fontbuffer=fitz.Font('tiro').buffer)
    with fitz.open(OUT/'dataset_legend.pdf') as src:
        page.show_pdf_page(fitz.Rect(0,0,width*72,.42*72),src,0)
    specs=[('figure2a_families',0,3.2,'(a) Defect families'),
           ('figure2b_area',3.30,2.5,'(b) Region area'),
           ('figure1',5.85,2.5,'(c) Defect multiplicity')]
    for name,x,w,title in specs:
        with fitz.open(OUT/f'{name}.pdf') as src:
            page.show_pdf_page(fitz.Rect(x*72,.42*72,(x+w)*72,2.77*72),src,0)
        result=page.insert_textbox(fitz.Rect(x*72,2.77*72,(x+w)*72,3.12*72),title,fontname='PanelTimes',fontsize=11,align=1)
        assert result>=0, f'Panel title clipped: {title}'
    doc.save(OUT/'dataset_composition.pdf',garbage=4,deflate=True)
    page.get_pixmap(matrix=fitz.Matrix(2,2)).save(OUT/'dataset_composition.png');doc.close()
    # Remove obsolete two-panel output; it is not part of the revised manuscript.
    for name in ['figure2.pdf','figure2.png']:
        (OUT/name).unlink(missing_ok=True)
    (ROOT/'data/figure_render_spec.json').write_text(json.dumps(SPECS,indent=2,default=lambda x:x.item())+'\n')
    print('Rendered one three-panel benchmark-composition figure from 1,983 regions and 718 flawed images.')

if __name__=='__main__':main()
