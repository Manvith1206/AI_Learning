from typing import Dict, List, Any, Callable, Set
from dataclasses import dataclass


@dataclass
class Event:
    """Base class for all events in the system"""
    name: str
    data: Dict[str, Any]


class EventBus:
    """Simple event bus for publishing and subscribing to events"""
    
    def __init__(self):
        self._subscribers: Dict[str, Set[Callable[[Event], None]]] = {}
    
    def subscribe(self, event_name: str, callback: Callable[[Event], None]) -> None:
        """Subscribe to an event"""
        if event_name not in self._subscribers:
            self._subscribers[event_name] = set()
        
        self._subscribers[event_name].add(callback)
    
    def unsubscribe(self, event_name: str, callback: Callable[[Event], None]) -> None:
        """Unsubscribe from an event"""
        if event_name in self._subscribers and callback in self._subscribers[event_name]:
            self._subscribers[event_name].remove(callback)
            
            # Clean up empty sets
            if not self._subscribers[event_name]:
                del self._subscribers[event_name]
    
    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers"""
        if event.name in self._subscribers:
            for callback in list(self._subscribers[event.name]):
                try:
                    callback(event)
                except Exception as e:
                    # In a real application, you'd want to log this error
                    print(f"Error in event handler for {event.name}: {e}")


# Singleton instance
event_bus = EventBus()
