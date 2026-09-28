"""Sample retained local mesh layers using the application's exact indices."""
import numpy as np


def transpose_mesh(mesh):
    """Swap both coordinates through every retained layer and its evidence."""
    rows,columns=mesh['rows'],mesh['columns']
    field=np.asarray(mesh['grid']).reshape(rows,columns,2)
    result={**mesh,'rows':columns,'columns':rows,
            'grid':field.transpose(1,0,2)[:,:,::-1].reshape(-1,2).tolist()}
    if 'grid_window' in mesh:
        nr,nc,ro,co=mesh['grid_window'];result['grid_window']=[nc,nr,co,ro]
    if 'refinement_regions' in mesh:
        result['refinement_regions']=[[y0,x0,y1,x1] for x0,y0,x1,y1 in mesh['refinement_regions']]
    if mesh.get('preserved_mesh') is not None:
        result['preserved_mesh']=transpose_mesh(mesh['preserved_mesh'])
    for name in ('established_rulers','established_text_rows','established_text_columns'):
        if name not in mesh:continue
        result[name]=[]
        for item in mesh[name]:
            swapped={**item,'points':np.asarray(item['points'])[:,::-1].tolist()}
            if 'reference' in item:swapped['reference']=np.asarray(item['reference'])[:,::-1].tolist()
            if item.get('normal_axis') is not None:swapped['normal_axis']=1-item['normal_axis']
            result[name].append(swapped)
    for name in ('retained_word_clouds','retained_word_references',
                 'unrecognized_print_clouds','unrecognized_print_references'):
        if name in mesh:
            result[name]=[np.asarray(group)[:,::-1].tolist() for group in mesh[name]]
    if 'complete_printed_strokes' in mesh:
        result['complete_printed_strokes']=[
            {**owner,'axis':1-owner['axis'],
             'source_points':np.asarray(owner['source_points'])[:,::-1].tolist(),
             'reference_points':np.asarray(owner['reference_points'])[:,::-1].tolist()}
            for owner in mesh['complete_printed_strokes']]
    if 'retained_pixel_footprints' in mesh:
        from material_pixel_footprints import transpose_owner
        result['retained_pixel_footprints']=[transpose_owner(owner)
            for owner in mesh['retained_pixel_footprints']]
    if 'tangent_free_rule_proxy_roles' in mesh:
        def swap_roles(ledger):
            if ledger is None:return None
            return {**ledger,'roles':[
                {**role,'normal_axis':1-role['normal_axis'],
                 'points':np.asarray(role['points'])[:,::-1].tolist(),
                 'source_points':np.asarray(role['source_points'])[:,::-1].tolist()}
                for role in ledger['roles']],
                'earlier_roles':swap_roles(ledger.get('earlier_roles'))}
        result['tangent_free_rule_proxy_roles']=swap_roles(mesh['tangent_free_rule_proxy_roles'])
    return result


class MeshSampler:
    def __init__(self, mesh):
        self.rows, self.columns = mesh['rows'], mesh['columns']
        self.grid = np.asarray(mesh['grid'], dtype=float).reshape(self.rows, self.columns, 2)
        self.window = mesh.get('grid_window')
        self.regions = mesh.get('refinement_regions', [])
        parent = mesh.get('preserved_mesh')
        self.parent = None if parent is None else MeshSampler(parent)
        self.coordinate_transform = mesh.get('coordinate_transform', False)
        if not isinstance(self.coordinate_transform, bool) or (self.coordinate_transform and self.parent is None):
            raise ValueError('Coordinate transform requires a valid parent')

    def sample(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('Mesh queries must be finite paired coordinates')
        if self.parent is None:
            active = np.ones(len(points), bool)
        else:
            active = np.zeros(len(points), bool)
            for x0, y0, x1, y1 in self.regions:
                active |= ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
                           (points[:, 1] >= y0) & (points[:, 1] <= y1))
        result = np.empty_like(points)
        if (~active).any():
            result[~active] = self.parent.sample(points[~active])
        query = points[active]
        if not len(query):
            return result
        if self.window is None:
            full_rows, full_columns, row_offset, column_offset = self.rows, self.columns, 0, 0
        else:
            full_rows, full_columns, row_offset, column_offset = self.window
        x = np.clip(query[:, 0] * (full_columns-1), 0, full_columns-1-1e-9) - column_offset
        y = np.clip(query[:, 1] * (full_rows-1), 0, full_rows-1-1e-9) - row_offset
        ix = np.clip(np.floor(x).astype(int), 0, self.columns-2)
        iy = np.clip(np.floor(y).astype(int), 0, self.rows-2)
        fx, fy = x-ix, y-iy
        corners = [self.grid[iy, ix], self.grid[iy, ix+1],
                   self.grid[iy+1, ix], self.grid[iy+1, ix+1]]
        weights = [(1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy]
        if self.coordinate_transform:
            for index, (dx, dy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
                uv = np.c_[(ix+column_offset+dx)/(full_columns-1),
                           (iy+row_offset+dy)/(full_rows-1)]
                corners[index] = corners[index] - uv
            mapped = query + sum(g*w[:, None] for g, w in zip(corners, weights))
            result[active] = self.parent.sample(mapped)
        else:
            result[active] = sum(g*w[:, None] for g, w in zip(corners, weights))
        return result

    def dense(self, rows, columns):
        result = np.empty((rows, columns, 2), dtype=float)
        for start in range(0, rows, 64):
            end = min(rows, start+64)
            x,y = np.meshgrid(np.linspace(0,1,columns), np.arange(start,end)/(rows-1))
            result[start:end] = self.sample(np.c_[x.ravel(),y.ravel()]).reshape(end-start,columns,2)
        return result
