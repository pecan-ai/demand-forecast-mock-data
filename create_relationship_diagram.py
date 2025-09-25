import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_relationship_diagram():
    """
    Create a visual diagram showing the relationships between tables.
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define table positions and sizes
    tables = {
        'entity': {'pos': (1, 8), 'size': (1.5, 0.8), 'color': '#FFE6E6'},
        'sku-colddirnks': {'pos': (1, 6.5), 'size': (1.5, 0.8), 'color': '#E6F3FF'},
        'entity-business-importance': {'pos': (1, 5), 'size': (1.5, 0.8), 'color': '#E6FFE6'},
        'segmentation-entity': {'pos': (1, 3.5), 'size': (1.5, 0.8), 'color': '#FFF0E6'},
        'segmentation': {'pos': (1, 2), 'size': (1.5, 0.8), 'color': '#F0E6FF'},
        'building-blocks': {'pos': (4, 8), 'size': (1.5, 0.8), 'color': '#E6FFFF'},
        'building-block-feature-map': {'pos': (4, 6.5), 'size': (1.5, 0.8), 'color': '#FFE6FF'},
        'live-predictions': {'pos': (7, 8), 'size': (1.5, 0.8), 'color': '#FFFFE6'},
        'residuals': {'pos': (7, 6.5), 'size': (1.5, 0.8), 'color': '#E6E6FF'},
        'stability': {'pos': (7, 5), 'size': (1.5, 0.8), 'color': '#FFE6E6'}
    }
    
    # Draw tables
    for table_name, props in tables.items():
        x, y = props['pos']
        width, height = props['size']
        color = props['color']
        
        # Create rounded rectangle for table
        table_box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(table_box)
        
        # Add table name
        ax.text(x + width/2, y + height/2, table_name.replace('-', '\n'), 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw relationships
    relationships = [
        # Entity relationships
        ('entity', 'sku-colddirnks', 'SKU_ID'),
        ('entity', 'entity-business-importance', 'Entity'),
        ('entity', 'segmentation-entity', 'Entity'),
        ('entity', 'live-predictions', 'Entity'),
        ('entity', 'residuals', 'Entity'),
        ('entity', 'stability', 'Entity'),
        
        # Segmentation relationships
        ('segmentation-entity', 'segmentation', 'segment_id'),
        
        # Building block relationships
        ('building-blocks', 'building-block-feature-map', 'name/building_block'),
        
        # Time series relationships
        ('live-predictions', 'residuals', 'Entity, Marker, Horizon'),
        ('residuals', 'stability', 'Entity, Cycle, Marker, Horizon')
    ]
    
    # Draw relationship lines
    for source, target, key in relationships:
        if source in tables and target in tables:
            source_pos = tables[source]['pos']
            target_pos = tables[target]['pos']
            
            # Calculate connection points
            if source_pos[0] < target_pos[0]:  # Source is to the left
                start_x = source_pos[0] + tables[source]['size'][0]
                start_y = source_pos[1] + tables[source]['size'][1]/2
                end_x = target_pos[0]
                end_y = target_pos[1] + tables[target]['size'][1]/2
            else:  # Source is to the right or same column
                start_x = source_pos[0]
                start_y = source_pos[1] + tables[source]['size'][1]/2
                end_x = target_pos[0] + tables[target]['size'][0]
                end_y = target_pos[1] + tables[target]['size'][1]/2
            
            # Draw arrow
            ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
            
            # Add relationship label
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            ax.text(mid_x, mid_y + 0.1, key, ha='center', va='bottom', 
                   fontsize=6, color='blue', fontweight='bold')
    
    # Add title and legend
    ax.text(5, 9.5, 'Database Schema Relationships', ha='center', va='center', 
           fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#FFE6E6', edgecolor='black', label='Core Entity Tables'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#E6F3FF', edgecolor='black', label='Product Tables'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#E6FFE6', edgecolor='black', label='Business Metrics'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#FFF0E6', edgecolor='black', label='Segmentation'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#E6FFFF', edgecolor='black', label='Building Blocks'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#FFFFE6', edgecolor='black', label='Time Series Data')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig('database_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure instead of showing it
    
    print("Relationship diagram saved as 'database_relationships.png'")

if __name__ == "__main__":
    create_relationship_diagram()
