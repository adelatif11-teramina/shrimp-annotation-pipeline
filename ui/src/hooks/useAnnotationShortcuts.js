import { useHotkeys } from 'react-hotkeys-hook';

const ENTITY_SHORTCUTS = [
  ['1', 'SPECIES'],
  ['2', 'PATHOGEN'],
  ['3', 'DISEASE'],
  ['4', 'CLINICAL_SYMPTOM'],
  ['5', 'PHENOTYPIC_TRAIT'],
  ['6', 'GENE'],
  ['7', 'CHEMICAL_COMPOUND'],
  ['8', 'TREATMENT'],
  ['9', 'LIFE_STAGE'],
  ['0', 'MEASUREMENT'],
];

const MODE_SHORTCUTS = [
  ['e', 'entity'],
  ['r', 'relation'],
  ['t', 'topic'],
];

function useAnnotationShortcuts({
  onAccept,
  onReject,
  onModify,
  onSkip,
  onSave,
  onNext,
  onSetMode,
  onSetEntityType,
  openGuidelines,
  openKeyboardHelp,
  closeKeyboardHelp,
}) {
  useHotkeys('ctrl+enter', onAccept, { preventDefault: true }, [onAccept]);
  useHotkeys('ctrl+r', onReject, { preventDefault: true }, [onReject]);
  useHotkeys('ctrl+m', onModify, { preventDefault: true }, [onModify]);
  useHotkeys('ctrl+shift+s', onSkip, { preventDefault: true }, [onSkip]);
  useHotkeys('ctrl+s', onSave, { preventDefault: true }, [onSave]);
  useHotkeys('ctrl+n', onNext, { preventDefault: true }, [onNext]);
  useHotkeys('escape', closeKeyboardHelp, { preventDefault: true }, [closeKeyboardHelp]);

  MODE_SHORTCUTS.forEach(([key, mode]) => {
    useHotkeys(key, () => onSetMode(mode), {}, [onSetMode, mode]);
  });

  ENTITY_SHORTCUTS.forEach(([key, type]) => {
    useHotkeys(key, () => onSetEntityType(type), {}, [onSetEntityType, type]);
  });

  useHotkeys('?', openKeyboardHelp, {}, [openKeyboardHelp]);
  useHotkeys('h', openKeyboardHelp, {}, [openKeyboardHelp]);
  useHotkeys('f1', openGuidelines, { preventDefault: true }, [openGuidelines]);
}

export default useAnnotationShortcuts;
