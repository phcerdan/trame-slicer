from __future__ import annotations

from collections.abc import Callable

from slicer import (
    vtkMRMLAbstractViewNode,
    vtkMRMLInteractionEventData,
    vtkMRMLNode,
)
from undo_stack.signal import Signal
from vtkmodules.vtkCommonCore import vtkCommand, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkRenderingCore import (
    vtkActor2D,
    vtkPolyDataMapper2D,
    vtkProp,
    vtkProperty2D,
    vtkRenderer,
)

from .scissors_effect_parameters import BrushInteractionMode
from .segment_modifier import SegmentModifier
from .segmentation_effect_pipeline import SegmentationEffectPipeline
from .segmentation_effect_scissors import SegmentationEffectScissors


class _LineDrawer:
    def __init__(self, color: tuple[float, float, float], visibility: bool, point_size: float, line_width: float):
        self.points = vtkPoints()
        self.lines = vtkCellArray()
        self.vertices = vtkCellArray()
        self.polydata = vtkPolyData()
        self.polydata.SetLines(self.lines)
        self.polydata.SetVerts(self.vertices)
        self.polydata.SetPoints(self.points)

        self.mapper = vtkPolyDataMapper2D()
        self.mapper.SetInputData(self.polydata)
        self.actor = vtkActor2D()
        self.actor.SetMapper(self.mapper)
        self.actor.SetVisibility(visibility)

        props = self.actor.GetProperty()
        props.SetColor(*color)
        props.SetPointSize(point_size)
        props.SetLineWidth(line_width)

    def reset(self):
        self.points.SetNumberOfPoints(0)
        self.vertices.Reset()
        self.lines.Reset()
        self.polydata.Modified()


class ScissorsPolygonBrush:
    """Display the scissors as 2D lines"""

    def __init__(self):
        super().__init__()
        self._line_drawer = _LineDrawer((1.0, 1.0, 0.0), False, 4.0, 2.0)

        # Preview line
        self._preview_line_drawer = _LineDrawer((1.0, 1.0, 0.0), False, 4.0, 2.0)
        self._preview_line_drawer.points.SetNumberOfPoints(2)
        self._preview_line_drawer.lines.InsertNextCell(2, [0, 1])
        self._preview_line_drawer.vertices.InsertNextCell(1, [1])

        # Closing line
        self._closing_line_drawer = _LineDrawer((0.8, 0.8, 0.0), False, 0.0, 2.0)
        self._closing_line_drawer.points.SetNumberOfPoints(2)
        self._closing_line_drawer.lines.InsertNextCell(2, [0, 1])
        self._closing_line_drawer.vertices.InsertNextCell(1, [1])

    def set_visibility(self, visible: bool):
        self._line_drawer.actor.SetVisibility(visible)
        self._preview_line_drawer.actor.SetVisibility(visible)
        self._closing_line_drawer.actor.SetVisibility(visible)

    def move_preview_point(self, x: int, y: int) -> None:
        self._preview_line_drawer.points.SetPoint(1, [float(x), float(y), 1.0])
        count = self._line_drawer.points.GetNumberOfPoints()
        if count == 0:
            self._preview_line_drawer.points.SetPoint(0, [float(x), float(y), 1.0])
        else:
            self._preview_line_drawer.points.SetPoint(0, self._line_drawer.points.GetPoint(count - 1))
        self._preview_line_drawer.lines.InsertNextCell(2, [0, 1])
        self._preview_line_drawer.points.Modified()

    def add_point(self, x: int, y: int) -> None:
        self._line_drawer.points.InsertNextPoint([float(x), float(y), 1.0])
        count = self._line_drawer.points.GetNumberOfPoints()
        if count > 1:
            self._line_drawer.lines.InsertNextCell(2, [count - 1, count - 2])
        if count >= 3:
            self._closing_line_drawer.points.SetPoint(0, self._line_drawer.points.GetPoint(0))
            self._closing_line_drawer.points.SetPoint(1, self._line_drawer.points.GetPoint(count - 1))
            self._closing_line_drawer.points.Modified()
        self._line_drawer.vertices.InsertNextCell(1, [count - 1])
        self._line_drawer.polydata.Modified()

    def reset(self) -> None:
        self._line_drawer.reset()
        self._closing_line_drawer.points.SetPoint(1, (0, 0, 0))
        self._closing_line_drawer.points.SetPoint(0, (0, 0, 0))

    @property
    def points(self) -> vtkPoints:
        return self._line_drawer.points

    def get_prop(self) -> vtkProp:
        """
        Return brush prop.
        Can be used to add or remove the brush from the renderer, configure rendering properties (visibility, color, ...)
        """
        return self._line_drawer.actor

    def get_preview_prop(self) -> vtkProp:
        return self._preview_line_drawer.actor

    def get_closing_prop(self) -> vtkProp:
        return self._closing_line_drawer.actor

    def get_property(self) -> vtkProperty2D:
        return self._line_drawer.actor.GetProperty()


class SegmentationScissorsWidget:
    """
    On slice view project 2D points on slice (world pos)
    On 3D view project 2D points on focal plane (world pos)
    """

    interaction_stopped = Signal(vtkPoints)

    def __init__(self) -> None:
        self._modifier: SegmentModifier | None = None
        self._view_node: vtkMRMLAbstractViewNode | None = None
        self._renderer: vtkRenderer | None = None
        self._brush = ScissorsPolygonBrush()
        self._brush_enabled = False
        self._painting = False

    def set_view_node(self, view_node):
        self._view_node = view_node

    def set_modifier(self, modifier: SegmentModifier):
        self._modifier = modifier

    def set_renderer(self, renderer):
        self.disable_brush()
        self._renderer = renderer

    def move_preview_point(self, x: int, y: int) -> None:
        self._brush.move_preview_point(x, y)

    def add_point(self, x: int, y: int) -> None:
        self._brush.add_point(x, y)

    def set_active(self, is_active: bool) -> None:
        if is_active:
            self.enable_brush()
        else:
            self.disable_brush()

    def enable_brush(self) -> None:
        if not self._renderer:
            return

        self._brush.set_visibility(True)
        self._brush_enabled = True
        self._renderer.AddViewProp(self._brush.get_prop())
        self._renderer.AddViewProp(self._brush.get_preview_prop())
        self._renderer.AddViewProp(self._brush.get_closing_prop())

    def disable_brush(self) -> None:
        if not self._renderer:
            return

        if self.is_painting():
            self.stop_painting()
        self._brush.set_visibility(False)
        self._brush_enabled = False
        self._renderer.RemoveViewProp(self._brush.get_prop())
        self._renderer.RemoveViewProp(self._brush.get_preview_prop())
        self._renderer.RemoveViewProp(self._brush.get_closing_prop())

    def is_brush_enabled(self) -> bool:
        return self._brush_enabled

    def start_painting(self, x: int, y: int) -> None:
        self._painting = True
        self.add_point(x, y)

    def stop_painting(self) -> None:
        self._painting = False
        # Reset brush before emitting signal to ensure reset is done
        points = vtkPoints()
        points.DeepCopy(self._brush.points)
        self._brush.reset()
        self.interaction_stopped.emit(points)

    def is_painting(self) -> bool:
        return self._painting


class SegmentationScissorsPipeline(SegmentationEffectPipeline[SegmentationEffectScissors]):
    def __init__(self) -> None:
        super().__init__()

        self.widget = SegmentationScissorsWidget()

        # Events we may consume and how we consume them
        self._supported_events: dict[int, Callable] = {
            int(vtkCommand.MouseMoveEvent): self._MouseMoved,
            int(vtkCommand.LeftButtonPressEvent): self._LeftButtonPressed,
            int(vtkCommand.LeftButtonReleaseEvent): self._LeftButtonReleased,
            int(vtkCommand.RightButtonPressEvent): self._RightButtonPressed,
        }

    @property
    def brush_interaction_mode(self) -> BrushInteractionMode:
        return self._effect.brush_interaction_mode

    def SetActive(self, isActive: bool):
        super().SetActive(isActive)
        self.widget.interaction_stopped.connect(self._OnWidgetInteractionStopped)
        self.widget.set_modifier(self._effect.modifier)
        self.widget.set_active(is_active=isActive)

    def _OnWidgetInteractionStopped(self, points: vtkPoints) -> None:
        if points.GetNumberOfPoints() == 0:
            return
        self._effect.apply_points_display_coordinates(points, self._view)

    def OnRendererAdded(self, renderer: vtkRenderer | None) -> None:
        self.widget.set_renderer(renderer)

    def OnRendererRemoved(self, _renderer: vtkRenderer) -> None:
        self.widget.set_renderer(None)

    def SetViewNode(self, viewNode: vtkMRMLAbstractViewNode) -> None:
        super().SetViewNode(viewNode)
        self.widget.set_view_node(viewNode)

    def SetDisplayNode(self, displayNode: vtkMRMLNode) -> None:
        super().SetDisplayNode(displayNode)

    def CanProcessInteractionEvent(self, eventData: vtkMRMLInteractionEventData) -> tuple[bool, float]:
        can_process = self.IsActive() and self.IsSupportedEvent(eventData)
        return can_process, 0.0

    def ProcessInteractionEvent(self, event_data: vtkMRMLInteractionEventData) -> bool:
        if event_data.GetType() not in self._supported_events:
            return False

        callback = self._supported_events.get(event_data.GetType())
        return callback(event_data) if callback is not None else False

    def _LeftButtonPressed(self, event_data: vtkMRMLInteractionEventData) -> bool:
        x, y = event_data.GetDisplayPosition()
        if self.brush_interaction_mode == BrushInteractionMode.CONTINUOUS:
            self.widget.start_painting(x, y)
        elif self.brush_interaction_mode == BrushInteractionMode.POINT_BY_POINT:
            if not self.widget.is_painting():
                self.widget.start_painting(x, y)
            else:
                self.widget.add_point(x, y)
            self.widget.move_preview_point(x, y)
        self.RequestRender()
        return True

    def _LeftButtonReleased(self, _event_data: vtkMRMLInteractionEventData) -> bool:
        if self.brush_interaction_mode == BrushInteractionMode.CONTINUOUS:
            self.widget.stop_painting()
            self.RequestRender()
            return True
        return False

    def _MouseMoved(self, event_data: vtkMRMLInteractionEventData) -> bool:
        x, y = event_data.GetDisplayPosition()
        if self.widget.is_painting() and self.brush_interaction_mode == BrushInteractionMode.CONTINUOUS:
            self.widget.add_point(x, y)
        self.widget.move_preview_point(x, y)
        self.RequestRender()

        # Always let other interactor and displayable managers do whatever they want
        return False

    def _RightButtonPressed(self, _event_data: vtkMRMLInteractionEventData) -> bool:
        if self.brush_interaction_mode == BrushInteractionMode.POINT_BY_POINT:
            self.widget.stop_painting()
            self.RequestRender()
            return True
        return False

    def IsSupportedEvent(self, event_data: vtkMRMLInteractionEventData):
        return event_data.GetType() in self._supported_events
