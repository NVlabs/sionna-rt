#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Sphinx extension: auto-generate ITU materials table.

Provides the ``itu-materials-table`` directive.
"""

from docutils import nodes
from docutils.parsers.rst import Directive

class ITUMaterialsTableDirective(Directive):
    """
    Based on the ITU Materials definitions implemented in Sionna RT, this directive
    generates a table listing each material and its properties.
    """

    def run(self):
        from sionna.rt.radio_materials.itu import ITU_MATERIALS_PROPERTIES
        from sionna.rt.radio_materials.itu_material import ITURadioMaterial

        table = nodes.table()
        tgroup = nodes.tgroup(cols=7)
        table += tgroup

        # Col specs
        for w in [20, 10, 10, 10, 15, 10, 20]:
            tgroup += nodes.colspec(colwidth=w)

        thead = nodes.thead()
        tgroup += thead

        # Header Row 1
        row1 = nodes.row()

        entry_mat = nodes.entry(morerows=1)
        entry_mat += nodes.paragraph('', 'Material type')
        row1 += entry_mat

        entry_color = nodes.entry(morerows=1)
        entry_color += nodes.paragraph('', 'Color')
        row1 += entry_color

        entry_perm = nodes.entry(morecols=1)
        entry_perm += nodes.paragraph('', 'Real part of relative permittivity')
        row1 += entry_perm

        entry_cond = nodes.entry(morecols=1)
        entry_cond += nodes.paragraph('', 'Conductivity [S/m]')
        row1 += entry_cond

        entry_freq = nodes.entry(morerows=1)
        entry_freq += nodes.paragraph('', 'Frequency range (GHz)')
        row1 += entry_freq

        thead += row1

        # Header Row 2
        row2 = nodes.row()
        for text in ['a', 'b', 'c', 'd']:
            entry = nodes.entry()
            entry += nodes.paragraph('', text)
            row2 += entry
        thead += row2

        tbody = nodes.tbody()
        tgroup += tbody

        def create_entry(text, morerows=0):
            entry = nodes.entry(morerows=morerows)
            if text.startswith(':math:`') and text.endswith('`'):
                math_text = text[7:-1]
                p = nodes.paragraph()
                p += nodes.math('', math_text)
                entry += p
            elif text.startswith('<span'):
                p = nodes.paragraph()
                p += nodes.raw('', text, format='html')
                entry += p
            else:
                p = nodes.paragraph('', text)
                entry += p
            return entry

        for name, ranges in ITU_MATERIALS_PROPERTIES.items():
            if name == "vacuum":
                color_col = ""
            else:
                color = ITURadioMaterial.ITU_MATERIAL_COLORS.get(name, (0.0, 0.0, 0.0))
                r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
                color_col = f'<span style="display:inline-block;width:15px;height:15px;background-color:rgb({r},{g},{b});border:1px solid #aaa;"></span>'

            num_ranges = len(ranges)

            for i, (f_range, params) in enumerate(ranges.items()):
                row = nodes.row()

                if i == 0:
                    row += create_entry(name, morerows=num_ranges-1)
                    row += create_entry(color_col, morerows=num_ranges-1)

                a, b, c, d = params

                if c == 0:
                    c_str = "0"
                elif c >= 0.0001:
                    c_str = str(c)
                else:
                    c_str = f"{c:g}"
                    if "e" in c_str:
                        base, exp = c_str.split("e")
                        exp_val = int(exp)
                        c_str = f":math:`{base} \\times 10^{{{exp_val}}}`"
                if c == 1e7:
                    c_str = r":math:`10^7`"

                if f_range[1] == 10.0 and name in ["very_dry_ground", "medium_dry_ground", "wet_ground"]:
                    freq_str = f"{f_range[0]:.1f} -- {f_range[1]:.0f} only"
                else:
                    if f_range[0] == 0.001:
                        freq_str = f"{f_range[0]} -- {f_range[1]:.0f}"
                    elif f_range[0] == 0.1:
                        freq_str = f"{f_range[0]} -- {f_range[1]:.0f}"
                    else:
                        freq_str = f"{f_range[0]:.0f} -- {f_range[1]:.0f}"

                row += create_entry(str(a))
                row += create_entry(str(b))
                row += create_entry(c_str)
                row += create_entry(str(d))
                row += create_entry(freq_str)

                tbody += row

        return [table]

def setup(app):
    app.add_directive("itu-materials-table", ITUMaterialsTableDirective)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
