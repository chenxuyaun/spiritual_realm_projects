"""
Structured Lesson Pack Data Models.

This module defines the data structures for structured lesson output
with JSON serialization support and validation.

Requirements:
    - 18.1: JSON-structured lesson output
    - 18.2: LessonSection with all required fields
    - 19.1: Minimum sections validation
    - 19.2: Content requirement validation
    - 19.4: Completeness score calculation
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
import json


@dataclass
class LessonSection:
    """
    Structured section of teaching content.
    
    A lesson section represents a distinct part of the teaching content
    with specific pedagogical elements.
    
    Attributes:
        name: Section name (e.g., "导入", "新授", "练习", "小结")
        teacher_say: What the teacher says/presents in this section
        student_may_say: Optional expected student responses
        examples: List of example problems or demonstrations
        questions: List of questions for students
        key_points: List of key takeaways from this section
        tips: Optional teaching tips or notes
    
    Requirements:
        - 18.2: LessonSection with all required fields
    """
    
    name: str
    teacher_say: str
    student_may_say: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    tips: Optional[str] = None
    
    def has_content(self) -> bool:
        """
        Check if section has meaningful content (examples or questions).
        
        Returns:
            True if section contains examples or questions
        
        Requirements:
            - 19.2: Content requirement validation
        """
        return len(self.examples) > 0 or len(self.questions) > 0
    
    def completeness_score(self) -> float:
        """
        Calculate completeness score for this section.
        
        Score is based on presence of optional fields:
        - examples: 0.25
        - questions: 0.25
        - key_points: 0.25
        - student_may_say: 0.15
        - tips: 0.10
        
        Returns:
            Completeness score between 0.0 and 1.0
        
        Requirements:
            - 19.4: Completeness score calculation
        """
        score = 0.0
        
        if self.examples:
            score += 0.25
        if self.questions:
            score += 0.25
        if self.key_points:
            score += 0.25
        if self.student_may_say:
            score += 0.15
        if self.tips:
            score += 0.10
        
        return score


@dataclass
class StructuredLesson:
    """
    Complete structured lesson plan with JSON serialization.
    
    A structured lesson contains metadata and a list of sections
    that together form a complete teaching unit.
    
    Attributes:
        topic: The lesson topic
        grade: Target grade level or difficulty
        sections: List of lesson sections
    
    Requirements:
        - 18.1: JSON-structured lesson output
        - 18.2: StructuredLesson with to_json and from_json methods
        - 19.1: Minimum sections validation
        - 19.2: Content requirement validation
        - 19.4: Completeness score calculation
    """
    
    topic: str
    grade: str
    sections: List[LessonSection] = field(default_factory=list)
    
    def to_json(self) -> Dict[str, Any]:
        """
        Serialize lesson to JSON-compatible dictionary.
        
        Returns:
            Dictionary with topic, grade, and sections
        
        Requirements:
            - 18.1: JSON-structured lesson output
            - 18.2: to_json method
        """
        return {
            "topic": self.topic,
            "grade": self.grade,
            "sections": [asdict(section) for section in self.sections]
        }
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "StructuredLesson":
        """
        Deserialize lesson from JSON-compatible dictionary.
        
        Args:
            data: Dictionary with topic, grade, and sections
        
        Returns:
            StructuredLesson instance
        
        Raises:
            ValueError: If required fields are missing
            TypeError: If data types are incorrect
        
        Requirements:
            - 18.2: from_json method
        """
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary")
        
        if "topic" not in data:
            raise ValueError("Missing required field: topic")
        if "grade" not in data:
            raise ValueError("Missing required field: grade")
        if "sections" not in data:
            raise ValueError("Missing required field: sections")
        
        if not isinstance(data["sections"], list):
            raise TypeError("Sections must be a list")
        
        # Parse sections
        sections = []
        for i, section_data in enumerate(data["sections"]):
            if not isinstance(section_data, dict):
                raise TypeError(f"Section {i} must be a dictionary")
            
            if "name" not in section_data:
                raise ValueError(f"Section {i} missing required field: name")
            if "teacher_say" not in section_data:
                raise ValueError(f"Section {i} missing required field: teacher_say")
            
            section = LessonSection(
                name=section_data["name"],
                teacher_say=section_data["teacher_say"],
                student_may_say=section_data.get("student_may_say"),
                examples=section_data.get("examples", []),
                questions=section_data.get("questions", []),
                key_points=section_data.get("key_points", []),
                tips=section_data.get("tips")
            )
            sections.append(section)
        
        return cls(
            topic=data["topic"],
            grade=data["grade"],
            sections=sections
        )
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate lesson structure and content.
        
        Checks:
        1. Minimum 3 sections requirement
        2. At least one section must have examples or questions
        
        Returns:
            Tuple of (is_valid, error_messages)
        
        Requirements:
            - 19.1: Minimum sections validation
            - 19.2: Content requirement validation
        """
        errors = []
        
        # Check minimum sections
        if len(self.sections) < 3:
            errors.append(
                f"Lesson must have at least 3 sections, found {len(self.sections)}"
            )
        
        # Check content requirement
        has_content = any(section.has_content() for section in self.sections)
        if not has_content:
            errors.append(
                "Lesson must contain at least one section with examples or questions"
            )
        
        return len(errors) == 0, errors
    
    def completeness_score(self) -> float:
        """
        Calculate overall completeness score for the lesson.
        
        The score is the average of all section completeness scores.
        Returns 0.0 if there are no sections.
        
        Returns:
            Completeness score between 0.0 and 1.0
        
        Requirements:
            - 19.4: Completeness score calculation
        """
        if not self.sections:
            return 0.0
        
        total_score = sum(section.completeness_score() for section in self.sections)
        return total_score / len(self.sections)
    
    def to_json_string(self, indent: int = 2) -> str:
        """
        Serialize lesson to JSON string.
        
        Args:
            indent: Number of spaces for indentation
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_json(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json_string(cls, json_str: str) -> "StructuredLesson":
        """
        Deserialize lesson from JSON string.
        
        Args:
            json_str: JSON string representation
        
        Returns:
            StructuredLesson instance
        
        Raises:
            ValueError: If JSON is invalid or required fields are missing
        """
        try:
            data = json.loads(json_str)
            return cls.from_json(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
